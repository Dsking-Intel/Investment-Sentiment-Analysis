import torch
import feedparser
import csv
import requests
import functools
import time
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['figure.autolayout'] = False

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
#from transformers import pipeline
import torch.nn.functional as F
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup


# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODELS
# ============================================================
import streamlit as st

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert"
    )

    finbert = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    ).to(device)

    embed_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embed_model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).to(device)

    return (
        tokenizer,
        finbert,
        embed_tokenizer,
        embed_model
    )

tokenizer, finbert, embed_tokenizer, embed_model = load_models()

# ============================================================
# FEED HEALTH CHECK
# ============================================================
def check_feed(url, source_name, is_json=False, json_key=None, timeout=5):
    start = time.time()
    status = "OK"
    headlines = []
    last_updated = None

    try:
        if is_json:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            latency = time.time() - start

            if response.status_code != 200:
                return {
                    "source": source_name,
                    "status": "FAIL",
                    "count": 0,
                    "latency": latency,
                    "last_updated": None,
                    "headlines": []
                }

            data = response.json()
            items = data.get(json_key, [])
            headlines = [f"[{source_name}] {item['body']}" for item in items]

            if items:
                ts = items[0].get("created_at")
                if ts:
                    last_updated = ts

        else:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            latency = time.time() - start

            if not feed.entries:
                status = "EMPTY"
            else:
                headlines = [f"[{source_name}] {entry.title}" for entry in feed.entries]

                if hasattr(feed.feed, "updated_parsed") and feed.feed.updated_parsed:
                    last_updated = datetime(*feed.feed.updated_parsed[:6], tzinfo=timezone.utc)

        if last_updated:
            age_hours = (datetime.now(timezone.utc) - last_updated).total_seconds() / 3600
            if age_hours > 24:
                status = "STALE"

        return {
            "source": source_name,
            "status": status,
            "count": len(headlines),
            "latency": latency,
            "last_updated": last_updated,
            "headlines": headlines
        }

    except Exception:
        latency = time.time() - start
        return {
            "source": source_name,
            "status": "FAIL",
            "count": 0,
            "latency": latency,
            "last_updated": None,
            "headlines": []
        }



# ============================================================
# RUN ALL FEED CHECKS
# ============================================================
def run_all_feed_checks():
    feeds = [


    # ============================
    # YAHOO FINANCE (TICKER RSS)
    # ============================
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^FTSE&region=UK&lang=en-UK", "Yahoo:FTSE"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOG&region=US&lang=en-US", "Yahoo:GOOG"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMZN&region=US&lang=en-US", "Yahoo:AMZN"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US", "Yahoo:NVDA"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US", "Yahoo:JPM"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=V&region=US&lang=en-US", "Yahoo:Visa"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=MA&region=US&lang=en-US", "Yahoo:Mastercard"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=XOM&region=US&lang=en-US", "Yahoo:Exxon"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^RUT&region=US&lang=en-US", "Yahoo:Russell2000"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US", "Yahoo:Gold"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL=F&region=US&lang=en-US", "Yahoo:CrudeOil"),

    # ============================
    # BLOOMBERG ALTERNATIVES
    # ============================
        ("https://www.bnnbloomberg.ca/polopoly_fs/1.3859769!/menu/rss.xml", "BNN Bloomberg"),
        ("https://www.bnnbloomberg.ca/polopoly_fs/1.2270499!/menu/rss.xml", "BNN Bloomberg Markets"),
        ("https://www.bnnbloomberg.ca/polopoly_fs/1.2270501!/menu/rss.xml", "BNN Bloomberg Investing"),

    # ============================
    # REUTERS VIA MSN MIRRORS
    # ============================
        ("https://www.msn.com/en-us/money/rss", "MSN Money (Reuters)"),
        ("https://www.msn.com/en-us/news/world/rss", "MSN World (Reuters)"),
        ("https://www.msn.com/en-us/news/business/rss", "MSN Business (Reuters)"),

    # ============================
    # CORE FINANCIAL NEWS
    # ============================
        ("https://www.marketwatch.com/rss/topstories", "MarketWatch"),
        ("https://www.investing.com/rss/news_25.rss", "Investing"),
        ("https://www.ft.com/?format=rss", "FT"),
        ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ"),
        ("http://feeds.bbci.co.uk/news/business/rss.xml", "BBC Business"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC"),
        ("https://seekingalpha.com/feed.xml", "SeekingAlpha"),

    # ============================
    # POLICY & REGULATORY SOURCES
    # ============================
        ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom", "SEC Filings"),
        ("https://www.federalreserve.gov/feeds/press_all.xml", "Federal Reserve"),
        ("https://www.ecb.europa.eu/rss/press.xml", "ECB"),
        ("https://www.opec.org/opec_web/en/press_rss.xml", "OPEC"),

    # ============================
    # SECTOR-SPECIFIC SOURCES
    # ============================
        ("https://techcrunch.com/feed/", "TechCrunch"),
        ("https://venturebeat.com/feed/", "VentureBeat"),
        ("https://oilprice.com/rss/main", "OilPrice"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
        ("https://cointelegraph.com/rss", "CoinTelegraph"),

    # ============================
    # GLOBAL MARKET SOURCES
    # ============================
        ("https://asia.nikkei.com/rss/feed/nar", "Nikkei Asia"),
        ("https://www.scmp.com/rss/91/feed", "SCMP Business"),
        ("https://www.theguardian.com/uk/business/rss", "Guardian Business"),

    # ============================
    # SOCIAL SENTIMENT (JSON)
    # ============================
        ("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json", "StockTwits:AAPL", True, "messages"),
        ("https://api.stocktwits.com/api/2/streams/symbol/TSLA.json", "StockTwits:TSLA", True, "messages"),
        ("https://api.stocktwits.com/api/2/streams/symbol/NVDA.json", "StockTwits:NVDA", True, "messages"),
        ("https://api.stocktwits.com/api/2/streams/symbol/SPY.json", "StockTwits:SPY", True, "messages"),

    # ============================
    # REDDIT (RSS MIRRORS)
    # ============================
        ("https://www.reddit.com/r/wallstreetbets/.rss", "Reddit:WSB"),
        ("https://www.reddit.com/r/investing/.rss", "Reddit:Investing"),
    ]

    diagnostics = []
    all_headlines = []

    # --- RSS + JSON feeds ---
    for feed in feeds:
        if len(feed) == 2:
            url, name = feed
            diag = check_feed(url, name)
        else:
            url, name, is_json, key = feed
            diag = check_feed(url, name, is_json=is_json, json_key=key)

        diagnostics.append(diag)
        all_headlines.extend(diag["headlines"])

    # --- SCRAPERS ---
    all_headlines.extend(fetch_barrons())
    all_headlines.extend(fetch_business_insider())
    all_headlines.extend(fetch_economist())
    all_headlines.extend(fetch_institutional_investor())

    return diagnostics, all_headlines


# ============================================================
# SCRAPPERS
# ============================================================

def fetch_barrons():
    url = "https://www.barrons.com/latest"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        headlines = []
        for h in soup.select("h3"):
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(f"[Barrons] {text}")

        return headlines

    except Exception:
        return []


def fetch_business_insider():
    url = "https://www.businessinsider.com/markets"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        headlines = []
        for h in soup.select("h2, h3"):
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(f"[BusinessInsider] {text}")

        return headlines

    except Exception:
        return []


def fetch_economist():
    url = "https://www.economist.com/finance-and-economics"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        headlines = []
        for h in soup.select("h3"):
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(f"[Economist] {text}")

        return headlines

    except Exception:
        return []


def fetch_institutional_investor():
    url = "https://www.institutionalinvestor.com/latest"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        headlines = []
        for h in soup.select("h2, h3"):
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(f"[InstitutionalInvestor] {text}")

        return headlines

    except Exception:
        return []


def fetch_techcrunch():
    url = "https://techcrunch.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[TechCrunch] {h.get_text(strip=True)}"
                for h in soup.select("h2, h3") if len(h.get_text(strip=True)) > 20]
    except Exception:
        return []

def fetch_venturebeat():
    url = "https://venturebeat.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[VentureBeat] {h.get_text(strip=True)}"
                for h in soup.select("h2, h3") if len(h.get_text(strip=True)) > 20]
    except Exception:
        return []

def fetch_oilprice():
    url = "https://oilprice.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[OilPrice] {h.get_text(strip=True)}"
                for h in soup.select("h2, h3") if len(h.get_text(strip=True)) > 20]
    except:
        return []

def fetch_prnews():
    url = "https://www.prnewswire.com/news-releases/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[PRNewswire] {h.get_text(strip=True)}"
                for h in soup.select("h3") if len(h.get_text(strip=True)) > 20]
    except Exception:
        return []

def fetch_nikkei():
    url = "https://asia.nikkei.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[NikkeiAsia] {h.get_text(strip=True)}"
                for h in soup.select("h3") if len(h.get_text(strip=True)) > 20]
    except Exception:
        return []

def fetch_scmp():
    url = "https://www.scmp.com/business"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return [f"[SCMP] {h.get_text(strip=True)}"
                for h in soup.select("h3") if len(h.get_text(strip=True)) > 20]
    except Exception:
        return []

    
# ============================================================
# SANITISE HEADLINES
# ============================================================
def sanitize_headlines(headlines):
    return [h.replace("$", "") for h in headlines]

# ============================================================
# REMOVE DUPLICATES
# ============================================================

def clean_headline_text(h):
    """
    Normalises headline text so duplicates from scrapers + RSS collapse properly.
    """
    # Remove source tags like [Barrons], [Yahoo:FTSE], etc.
    h = re.sub(r"\[[^\]]+\]\s*", "", h)

    # Lowercase for comparison
    h = h.lower().strip()

    # Remove punctuation noise
    h = re.sub(r"[^\w\s]", "", h)

    # Collapse multiple spaces
    h = re.sub(r"\s+", " ", h)

    return h


def dedupe_headlines(headlines):
    """
    Removes duplicates across RSS, JSON, and scrapers.
    """
    seen = set()
    unique = []

    for h in headlines:
        cleaned = clean_headline_text(h)

        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(h)

    return unique


# ============================================================
# PRINT DIAGNOSTICS
# ============================================================
def print_feed_diagnostics(diags):
    print("\n================ NEWS FEED HEALTH CHECK ================\n")
    for d in diags:
        print(f"{d['source']}: {d['status']}")
        print(f"  Headlines: {d['count']}")
        print(f"  Latency: {d['latency']:.2f}s")
        if d["last_updated"]:
            print(f"  Last Updated: {d['last_updated']}")
        print("")

# ============================================================
# EMBEDDING (MEAN POOLING)
# ============================================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)


def compute_embeddings(headlines, batch_size=64):
    emb_dict = {}
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        print(type(embed_tokenizer))
        print(embed_tokenizer)
        inputs = embed_tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.no_grad():
            output = embed_model(**inputs)
            emb = mean_pooling(output, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        for h, e in zip(batch, emb):
            emb_dict[h] = e.cpu().numpy()

    return emb_dict

# ============================================================
# FINBERT SENTIMENT
# ============================================================
def analyze_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = finbert(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    idx = torch.argmax(probs).item()
    return {"label": LABELS[idx], "confidence": float(probs[0][idx])}

def compute_sentiments(headlines, batch_size=64):
    results = {}
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = finbert(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        for h, p in zip(batch, probs):
            idx = torch.argmax(p).item()
            results[h] = {
                "label": LABELS[idx],
                "confidence": float(p[idx]),
                "negative_prob": float(p[0]),
                "neutral_prob": float(p[1]),
                "positive_prob": float(p[2])
            }

    return results

# ============================================================
# OPTIMISED KEYWORD EXTRACTION (SAFE INDEXING)
# ============================================================

global_vectorizer = None

def build_global_keyword_matrix(headlines):
    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words="english")
    clean = [re.sub(r"\b\d+(\.\d+)?%?\b", "", h) for h in headlines]
    matrix = vectorizer.fit_transform(clean)
    index_map = {h: i for i, h in enumerate(headlines)}
    return matrix, clean, index_map, vectorizer

def extract_cluster_keywords(cluster_headlines, global_matrix, index_map, top_n=5):
    indices = [index_map[h] for h in cluster_headlines if h in index_map]

    if not indices:
        return ["misc"]

    cluster_counts = global_matrix[indices].sum(axis=0).A1
    vocab = global_vectorizer.get_feature_names_out()

    pairs = sorted(zip(vocab, cluster_counts), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:top_n]]


# ============================================================
# SEMANTIC CLUSTER NAMING (EMBEDDING-BASED)
# ============================================================

semantic_topics = {
    "Technology": [
        "artificial intelligence", "amd","chips","intel","tsmc","saas","software", "cloud computing", "semiconductors","tech companies", "technology","digital transformation","machine learning","semiconductors", "chips", "AI", "Nvidia", "cloud", "data centers","software earnings", "tech stocks", "Silicon Valley"
    ],
    "Energy": [
    "oil markets", "energy production", "wind", "hydrogen", "renewables","gas prices", "crude oil", "energy companies", "solar", "turbine","exxon", "brent", "shell", "bp", "opec", "chevron", "wti","aramco", "petrochina"
    ],
    "Banking & Finance": [
        "financial markets", "consumer credit","banking sector", "credit risk", "assets", "liabilities", "capital", "exposure", "bank", "finance", "loan", "credit", "investment", "equity", "stock", "jpmorgan", "barclays", "hsbc"
    ],
    "Macro / Economy": [
        "inflation", "economic growth", "central banks", "monetary policy","GDP","CPI", "PPI", "GDP","central banks", "federal reserve", "ecb", "bank of england",
"monetary policy", "interest rate hikes", "economic indicators","labour market", "employment data", "recession risk"

    ],
    "Corporate Activity": [
        "earnings reports", "mergers and acquisitions", "corporate restructuring","dividends", "buybacks", "IPO activity"
    ],
    "Commodities": [
        "gold prices", "metals markets", "commodity trading", "silver", "copper", "mining companies", "glencore","rio tinto"
    ],
    "Crypto": [
        "bitcoin", "cryptocurrency markets", "blockchain", "digital assets","blockchain", "defi", "stablecoins", "crypto exchanges","binance", "coinbase", "crypto regulation"
    ],
    "Miscellaneous": [
        "general news", "miscellaneous topics", "other"
    ]
}


# ============================================================
# CLUSTER NAMING
# ============================================================
def name_clusters(clusters, emb_cache):
    named = {}
    for cid, items in clusters.items():
        named[cid] = name_cluster_semantic(items, emb_cache)
    return named

@functools.lru_cache(maxsize=1)
def compute_topic_embeddings():
    topic_embs = {}
    for topic, phrases in semantic_topics.items():
        inputs = embed_tokenizer(
            phrases,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            output = embed_model(**inputs)
            emb = mean_pooling(output, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        topic_embs[topic] = emb.mean(dim=0).cpu().numpy()

    return topic_embs

topic_embeddings = {}


def compute_cluster_centroid(cluster_headlines, emb_cache):
    vecs = [emb_cache[h] for h in cluster_headlines]

    centroid = np.mean(vecs, axis=0)

    norm = np.linalg.norm(centroid)

    if norm > 0:
        centroid = centroid / norm

    return centroid

def name_cluster_semantic(cluster_headlines, emb_cache):
    centroid = compute_cluster_centroid(cluster_headlines, emb_cache)

    best_topic = "Miscellaneous"
    best_score = -1

    for topic, topic_emb in topic_embeddings.items():
        score = float(np.dot(centroid, topic_emb))
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic



# ============================================================
# CLUSTERING (AUTO K SELECTION)
# ============================================================
def choose_optimal_k(vectors, k_min=3, k_max=10):
    if len(vectors) <= k_min:
        return max(1, len(vectors))

    best_k = k_min
    best_score = -1
    for k in range(k_min, min(k_max, len(vectors))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)
        score = silhouette_score(vectors, km.labels_)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

# ============================================================
# RELEVANCE
# ============================================================
def compute_relevance(question_emb, headline_emb):
    return float(F.cosine_similarity(
        torch.tensor(question_emb).unsqueeze(0),
        torch.tensor(headline_emb).unsqueeze(0)
    ))

# ============================================================
# CONTRADICTION DETECTION
# ============================================================
def sentiment_distance(q_label, h_label):
    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    return abs(mapping[q_label] - mapping[h_label])

def detect_contradiction(question_label, headline_label, relevance, threshold=0.20):
    if relevance < threshold:
        return "not relevant"

    dist = sentiment_distance(question_label, headline_label)
    if dist == 2:
        return "contradiction"
    if dist == 0:
        return "supports"
    return "neutral"

# ============================================================
# ANALYSIS PIPELINE
# ============================================================
def analyse_all(question, headlines, cluster_map, cluster_names, emb_cache, sentiment_cache):
    results = []
    q_sent = analyze_financial_sentiment(question)
    question_label = q_sent["label"]
    question_emb = compute_embeddings([question])[question]

    for h in headlines:
        sent = sentiment_cache[h]
        cosine = compute_relevance(question_emb, emb_cache[h])
        relevance = cosine

        contradiction = detect_contradiction(question_label, sent["label"], relevance)

        results.append({
            "text": h,
            "label": sent["label"],
            "confidence": sent["confidence"],
            "feedback": generate_feedback(sent["label"], sent["confidence"]),
            "relevance": relevance,
            "positive_prob": sent["positive_prob"],
            "neutral_prob": sent["neutral_prob"],
            "negative_prob": sent["negative_prob"],
            "contradiction": contradiction,
            "cluster": cluster_map[h],
            "cluster_name": cluster_names[cluster_map[h]],
            "error": ""
        })

    return results

# ============================================================
# FEEDBACK
# ============================================================
def generate_feedback(label, confidence):
    if label == "positive":
        return "This headline expresses optimism or favourable financial expectations."
    if label == "neutral":
        return "This headline is informational without clear positive or negative sentiment."
    if label == "negative":
        return "This headline reflects concern, risk, or unfavourable financial outlook."
    return "No sentiment detected."

# ============================================================
# CSV EXPORT
# ============================================================
def export_results_to_csv(results, filename):
    fieldnames = ["text","label","confidence","feedback","relevance","positive_prob","neutral_prob","negative_prob","contradiction","cluster","cluster_name","error"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)



# ============================================================
# STRIP DOLLAR SIGNS
# ============================================================
def _strip_dollar_signs(fig):
    for ax in fig.get_axes():
        for artist in ax.get_children():
            if hasattr(artist, "get_text"):
                txt = artist.get_text()
                if txt and "$" in txt:
                    artist.set_text(txt.replace("$", ""))

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            txt = label.get_text()
            if txt and "$" in txt:
                label.set_text(txt.replace("$", ""))

    for cbar in fig.get_children():
        if hasattr(cbar, "ax"):
            for label in cbar.ax.get_yticklabels():
                txt = label.get_text()
                if txt and "$" in txt:
                    label.set_text(txt.replace("$", ""))

    if hasattr(fig, "texts"):
        for t in fig.texts:
            txt = t.get_text()
            if txt and "$" in txt:
                t.set_text(txt.replace("$", ""))

# ============================================================
# CLUSTER SENTIMENT HEATMAP
# ============================================================
def plot_cluster_sentiment_heatmap(results, cluster_names):
    cluster_sentiment = {}

    for r in results:
        cid = r["cluster"]
        label = r["label"]
        if cid not in cluster_sentiment:
            cluster_sentiment[cid] = {"positive": 0, "neutral": 0, "negative": 0}
        cluster_sentiment[cid][label] += 1

    clusters = sorted(cluster_sentiment.keys())
    sentiments = ["positive", "neutral", "negative"]

    matrix = []
    for cid in clusters:
        matrix.append([cluster_sentiment[cid][s] for s in sentiments])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    heatmap = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    fig.colorbar(heatmap, ax=ax)

    ax.set_xticks(range(len(sentiments)))
    ax.set_xticklabels(sentiments)

    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([cluster_names[cid] for cid in clusters])

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Cluster")
    ax.set_title("Cluster Sentiment Heatmap")

    _strip_dollar_signs(fig)
    fig.tight_layout()
    return fig


# ============================================================
# RELEVANCE vs SENTIMENT SCATTER
# ============================================================

def plot_relevance_vs_sentiment(results, cluster_colors):
    relevance = [float(r["relevance"]) for r in results]
    confidence = [float(r["confidence"]) for r in results]
    colors = [cluster_colors[r["cluster"]] for r in results]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    ax.scatter(relevance, confidence, alpha=0.7, c=colors)

    ax.set_xlabel("Relevance Score")
    ax.set_ylabel("Sentiment Confidence")
    ax.set_title("Relevance vs Sentiment Confidence")

    sentiment_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Positive Cluster', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Neutral Cluster', markerfacecolor='grey', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Negative Cluster', markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=sentiment_handles, title="Cluster Sentiment", loc="lower left")

    fig.tight_layout()
    return fig

# ============================================================
# CLUSTER SENTIMENT DISTRIBUTION (STACKED BAR)
# ============================================================
def plot_cluster_sentiment_distribution(results, cluster_names):
    cluster_sentiment = {}

    for r in results:
        cid = r["cluster"]
        label = r["label"]
        if cid not in cluster_sentiment:
            cluster_sentiment[cid] = {"positive": 0, "neutral": 0, "negative": 0}
        cluster_sentiment[cid][label] += 1

    clusters = sorted(cluster_sentiment.keys())

    positive_counts = [cluster_sentiment[c]["positive"] for c in clusters]
    neutral_counts  = [cluster_sentiment[c]["neutral"]  for c in clusters]
    negative_counts = [cluster_sentiment[c]["negative"] for c in clusters]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    ax.bar(clusters, positive_counts, label="Positive", color="green")
    ax.bar(clusters, neutral_counts, bottom=positive_counts, label="Neutral", color="grey")
    ax.bar(clusters, negative_counts,
           bottom=[positive_counts[i] + neutral_counts[i] for i in range(len(clusters))],
           label="Negative", color="red")

    ax.set_xticks(clusters)
    ax.set_xticklabels([cluster_names[c] for c in clusters], rotation=45, ha="right")

    ax.set_ylabel("Headline Count")
    ax.set_title("Cluster Sentiment Distribution")
    ax.legend()

    fig.tight_layout()
    return fig


# ============================================================
# CLUSTER RELEVANCE BOXPLOT
# ============================================================
def plot_cluster_relevance_boxplot(results, cluster_names):
    cluster_relevance = {}

    for r in results:
        cid = r["cluster"]
        cluster_relevance.setdefault(cid, []).append(float(r["relevance"]))

    clusters = sorted(cluster_relevance.keys())
    data = [cluster_relevance[c] for c in clusters]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(data, patch_artist=True)

    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)

    ax.set_xticks(range(1, len(clusters) + 1))
    ax.set_xticklabels([cluster_names[c] for c in clusters], rotation=45, ha="right")

    ax.set_ylabel("Relevance Score")
    ax.set_title("Cluster Relevance Distribution")

    fig.tight_layout()
    return fig


# ============================================================
# KEYWORD CLOUDS
# ============================================================
def plot_cluster_keyword_cloud(clusters, cluster_names):
    fig, axes = plt.subplots(len(clusters), 1, figsize=(10, 5 * len(clusters)))

    if len(clusters) == 1:
        axes = [axes]

    for ax, (cid, headlines) in zip(axes, clusters.items()):
        text = " ".join([re.sub(r"\[.*?\]", "", h) for h in headlines])

        wc = WordCloud(width=800, height=400, background_color="white").generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{cluster_names[cid]} — Keyword Cloud")

    fig.tight_layout()
    return fig

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def generate_executive_summary(results, clusters, cluster_names):

    total_headlines = len(results)

    positive = sum(
        1 for r in results
        if r["label"] == "positive"
    )

    neutral = sum(
        1 for r in results
        if r["label"] == "neutral"
    )

    negative = sum(
        1 for r in results
        if r["label"] == "negative"
    )

    overall_sentiment = max(
        {
            "Positive": positive,
            "Neutral": neutral,
            "Negative": negative
        },
        key=lambda x: {
            "Positive": positive,
            "Neutral": neutral,
            "Negative": negative
        }[x]
    )

    largest_cluster_id = max(
        clusters,
        key=lambda c: len(clusters[c])
    )

    largest_cluster = cluster_names[largest_cluster_id]

    strongest_signals = sorted(
        results,
        key=lambda x: x["positive_prob"] * x["relevance"],
        reverse=True
    )[:3]

    biggest_risks = sorted(
        results,
        key=lambda x: x["negative_prob"] * x["relevance"],
        reverse=True
    )[:3]

    most_relevant = sorted(
        results,
        key=lambda x: x["relevance"],
        reverse=True
    )[:3]

    return {
        "total_headlines": total_headlines,
        "overall_sentiment": overall_sentiment,
        "largest_cluster": largest_cluster,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "strongest_signals": strongest_signals,
        "biggest_risks": biggest_risks,
        "most_relevant": most_relevant
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    question = input("Enter your financial question: ")

    diagnostics, combined = run_all_feed_checks()
    print_feed_diagnostics(diagnostics)

    # --- Deduplication FIRST ---
    combined = dedupe_headlines(combined)

    # --- Sanitise ---
    combined = sanitize_headlines(combined)

    print(f"Fetched {len(combined)} unique headlines after dedupe + sanitisation.")

    # --- Embeddings ---
    emb_cache = compute_embeddings(combined)

    # --- Sentiment ---
    sentiment_cache = compute_sentiments(combined)

    vectors = list(emb_cache.values())
    if len(vectors) == 0:
        print("No headlines fetched.")
        exit(0)

    # --- Clustering ---
    global_matrix, cleaned_headlines, index_map, global_vectorizer = build_global_keyword_matrix(combined)

    n_clusters = choose_optimal_k(vectors, k_min=3, k_max=7)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(n_clusters)}
    for h, label in zip(combined, labels):
        clusters[label].append(h)

    cluster_map = {h: label for h, label in zip(combined, labels)}

    global topic_embeddings
    topic_embeddings = compute_topic_embeddings()
    
    cluster_names = name_clusters(clusters, emb_cache)


    results = analyse_all(question, combined, cluster_map, cluster_names, emb_cache, sentiment_cache)
    ranked = sorted(results, key=lambda x: x["relevance"], reverse=True)

    print("\n================ TOP RELEVANT HEADLINES ================\n")
    for r in ranked[:10]:
        print(f"\nHeadline: {r['text']}")
        print(f"Relevance: {r['relevance']:.4f}")
        print(f"Sentiment: {r['label']} ({r['confidence']:.4f})")
        print(f"Feedback: {r['feedback']}")

    print("\n================ TOPIC CLUSTERS ================\n")
    for cid, items in clusters.items():
        print(f"\nCluster {cid} — {cluster_names[cid]}")
        for h in items[:10]:
            print(f"  - {h}")

    print("\n================ CONTRADICTORY HEADLINES ================\n")
    for r in results:
        if r["contradiction"] == "contradiction":
            print(f"  - {r['text']} (Relevance {r['relevance']:.3f})")

    plot_cluster_sentiment_heatmap(results, cluster_names)
  
    
