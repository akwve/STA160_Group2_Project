# Web scraping functions
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
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_content(text):
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

def scrape_article(url):
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
        "article",
        "main",
        "div[class*='article']",
        "div[class*='content']",
        "div[itemprop='articleBody']",
        "section[name='articleBody']",
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

def clean_text(text):
    """Remove irrelevant or boilerplate lines commonly found in news sites."""
    bad_phrases = [
        "news alerts",
        "breaking news",
        "there are no new alerts",
        "©",
        "sign up",
        "subscribe",
        "follow us",
        "advertisement",
        "read more",
        "all rights reserved",
        "supported by"
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

def scrape_and_clean_article(url):
    """Main function to scrape and clean article from URL."""
    title, body = scrape_article(url)
    all_text = title + ' ' + body
    cleaned_text = clean_content(all_text)
    return cleaned_text