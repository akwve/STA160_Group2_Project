"""
URL-based Fake News Detector with Gradio Interface
Run this script to launch a web interface that accepts news article URLs
"""

import gradio as gr
import joblib
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import quote
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
print("Checking NLTK data...")
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource, quiet=True)

print("Loading models...")
# Load the saved model and vectorizer
model = joblib.load("xgboost_fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("Models loaded successfully!")

def clean_content(text):
    """Clean and preprocess text with NLTK."""
    text = str(text).strip().lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

def wayback_fetch(url, headers):
    """Try to fetch the closest archived snapshot from the Wayback Machine."""
    api = f"https://archive.org/wayback/available?url={quote(url)}"
    r = requests.get(api, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    snap = data.get("archived_snapshots", {}).get("closest")
    if not snap or not snap.get("available"):
        raise requests.exceptions.RequestException("No archived snapshot available")
    snapshot_url = snap.get("url")
    rs = requests.get(snapshot_url, headers=headers, timeout=15)
    rs.raise_for_status()
    return BeautifulSoup(rs.text, "html.parser")

def clean_text(text):
    """Remove irrelevant or boilerplate lines commonly found in news sites."""
    bad_phrases = [
        "news alerts", "breaking news", "there are no new alerts",
        "©", "sign up", "subscribe", "follow us", "advertisement",
        "read more", "all rights reserved", "supported by"
    ]
    
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if any(bad in lower_line for bad in bad_phrases):
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def scrape_article(url):
    """Scrape article from URL with multiple fallback strategies."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # fetch website
    soup = None
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
    except requests.exceptions.RequestException:
        try:
            soup = wayback_fetch(url, headers)
        except requests.exceptions.RequestException:
            raise RuntimeError(f"Could not access the URL. Please try another link.")

    # try to detect JSON data (for sites like NBC, CNN, etc.)
    script_tag = soup.find("script", text=re.compile(r"window\.__data\s*="))
    if script_tag and script_tag.string:
        try:
            json_text = re.search(r"window\.__data\s*=\s*({.*});", script_tag.string).group(1)
            data = json.loads(json_text)
            title = data.get("initialState", {}).get("video", {}).get("currentVideo", {}).get("headline")
            body = (
                data.get("initialState", {}).get("video", {}).get("currentVideo", {}).get("description", "")
                + "\n"
                + data.get("initialState", {}).get("video", {}).get("currentVideo", {}).get("transcript", "")
            )
            if title and body:
                return title, clean_text(body)
        except Exception:
            pass

    # find title using common tags
    title_candidates = [
        soup.find("h1"),
        soup.find("meta", property="og:title"),
        soup.find("meta", attrs={"name": "twitter:title"}),
    ]
    title = next(
        (t.get_text(strip=True) if hasattr(t, "get_text") else t["content"])
        for t in title_candidates
        if t
    ) if any(title_candidates) else "(No title found)"

    # collect paragraphs from likely article sections
    body_candidates = [
        "article", "main",
        "div[class*='article']", "div[class*='content']",
        "div[itemprop='articleBody']", "section[name='articleBody']",
    ]

    paragraphs = []
    for selector in body_candidates:
        elements = soup.select(selector + " p")
        if elements:
            paragraphs = elements
            break

    if not paragraphs:
        paragraphs = soup.find_all("p")

    body_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
    return title, clean_text(body_text)

def scrape_and_clean_article(url):
    """Main function to scrape and clean article from URL."""
    title, body = scrape_article(url)
    all_text = title + ' ' + body
    cleaned_text = clean_content(all_text)
    return title, cleaned_text

def predict_news_from_url(url):
    """Scrape article from URL and return model prediction."""
    if not url.strip():
        return "⚠️ Please enter a URL.", "", ""
    
    try:
        # Scrape and clean the article
        title, cleaned_text = scrape_and_clean_article(url)
        
        if not cleaned_text.strip():
            return "⚠️ Could not extract meaningful text from the URL.", "", ""
        
        # Convert text to TF-IDF vector
        X_input = vectorizer.transform([cleaned_text])
        
        # Predict
        pred = model.predict(X_input)[0]
        
        # Get prediction probability for confidence score
        pred_proba = model.predict_proba(X_input)[0]
        confidence = max(pred_proba) * 100
        
        # XGBoost outputs 0/1, so map to labels
        result = "🟢 REAL NEWS" if pred == 1 else "🔴 FAKE NEWS"
        result_with_confidence = f"{result}\n\nConfidence: {confidence:.2f}%"
        
        # Return prediction, title, and a preview of the scraped text
        preview = cleaned_text[:800] + "..." if len(cleaned_text) > 800 else cleaned_text
        
        return result_with_confidence, f"📰 {title}", f"Article Preview (cleaned & preprocessed):\n\n{preview}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# Create Gradio interface with custom styling
with gr.Blocks(title="Fake News Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📰 Fake News Detector (URL-based)
        
        Enter a news article URL below. The system will:
        1. 🔍 Scrape the article content
        2. 🧹 Clean and preprocess the text
        3. 🤖 Analyze it using XGBoost ML model
        4. ✅ Classify it as Real or Fake news
        
        **Powered by:** XGBoost + TF-IDF + Advanced Web Scraping
        """
    )
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(
                label="🔗 News Article URL",
                placeholder="https://www.example.com/news/article",
                lines=2
            )
            submit_btn = gr.Button("🚀 Analyze Article", variant="primary", size="lg")
            
            gr.Markdown("### Example URLs to Try:")
            gr.Examples(
                examples=[
                    ["https://www.bbc.com/news/world-us-canada-67380199"],
                    ["https://www.reuters.com/world/"],
                    ["https://www.theguardian.com/world"],
                ],
                inputs=url_input
            )
    
    with gr.Row():
        prediction_output = gr.Textbox(label="🎯 Prediction Result", lines=3)
    
    with gr.Row():
        title_output = gr.Textbox(label="📋 Article Title", lines=2)
    
    with gr.Row():
        preview_output = gr.Textbox(label="📝 Scraped Content Preview", lines=15)
    
    submit_btn.click(
        fn=predict_news_from_url,
        inputs=url_input,
        outputs=[prediction_output, title_output, preview_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Fake News Detector Web Interface...")
    print("="*60 + "\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
