"""
Stock Picker app. Run with:  streamlit run app.py
Install: pip install streamlit yfinance feedparser vaderSentiment trafilatura requests pandas
Ranks a list of stocks and tells you which to BUY and which to AVOID.
Not financial advice.
"""
import re
from html import escape
from statistics import mean, pstdev

import feedparser
import pandas as pd
import requests
import streamlit as st
import trafilatura
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Picker", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero h1 {font-size: 2.3rem; margin: 0; letter-spacing: -0.02em;}
.hero p {color: #8b95a7; margin: .3rem 0 1.2rem;}
.pick {background: linear-gradient(135deg, #0f3d2e, #16664a); border: 1px solid #2ecc8f55;
       border-radius: 18px; padding: 18px 22px; margin: 8px 0 18px;}
.pick small {color: #9ad9bf; text-transform: uppercase; letter-spacing: .1em; font-size: .7rem;}
.pick .t {font-size: 1.8rem; font-weight: 800;}
.pick .n {color: #bfe9d8; font-size: .9rem;}
.card {background: #161b26; border: 1px solid #232a3a; border-left: 5px solid var(--c);
       border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;}
.card .top {display: flex; justify-content: space-between; align-items: center;}
.card .tk {font-weight: 800; font-size: 1.15rem;}
.card .nm {color: #8b95a7; font-size: .8rem;}
.pill {background: var(--c); color: #0b0f17; font-weight: 800; border-radius: 999px;
       padding: 3px 12px; font-size: .9rem;}
.chips span {display: inline-block; background: #202839; color: #c9d3e6; border-radius: 8px;
             padding: 3px 9px; font-size: .75rem; margin: 8px 6px 0 0;}
div[data-testid="stMetric"] {background: #161b26; border: 1px solid #232a3a;
                             border-radius: 14px; padding: 10px 16px;}
</style>
""", unsafe_allow_html=True)
vader = SentimentIntensityAnalyzer()


# ---------- Data ----------
def score_text(text):
    sents = re.split(r"(?<=[.!?])\s+", text)
    vals = [vader.polarity_scores(s)["compound"] for s in sents if len(s) > 20]
    return mean(vals) if vals else 0.0


@st.cache_data(ttl=900, show_spinner=False)
def fetch_text(url, ua="Mozilla/5.0"):
    try:
        html = requests.get(url, headers={"User-Agent": ua}, timeout=15).text
        return trafilatura.extract(html) or ""
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def get_articles(ticker, full_text, limit):
    rows = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    for it in items[:limit]:
        c = it.get("content", it)
        title = c.get("title", "")
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link", "")
        body = c.get("summary", "") or ""
        if full_text and url:
            body = fetch_text(url) or body
        if title:
            rows.append({"type": "Yahoo article", "title": title, "url": url,
                         "score": score_text(f"{title}. {body}")})
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock+when:14d")
    for e in feed.entries[:limit]:
        rows.append({"type": "Headline", "title": e.title, "url": e.link,
                     "score": vader.polarity_scores(e.title)["compound"]})
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def get_filings(ticker, email, limit=3):
    """Recent SEC 8-K filings (official company announcements)."""
    ua = f"StockPickerApp {email}"
    try:
        h = {"User-Agent": ua}
        m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=h, timeout=15).json()
        cik = next(v["cik_str"] for v in m.values() if v["ticker"] == ticker)
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=h, timeout=15)
        recent = r.json()["filings"]["recent"]
    except Exception:
        return []
    rows = []
    for form, acc, doc, date in zip(recent["form"], recent["accessionNumber"],
                                    recent["primaryDocument"], recent["filingDate"]):
        if form != "8-K":
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        text = fetch_text(url, ua)
        rows.append({"type": "SEC 8-K", "title": f"8-K filed {date}", "url": url,
                     "score": score_text(text[:8000]) if text else 0.0})
        if len(rows) >= limit:
            break
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def get_prices(ticker, days):
    return yf.Ticker(ticker).history(period=f"{days}d")["Close"]


def analyze(t, full_text, limit, email, days):
    try:
        p = get_prices(t, days)
    except Exception:
        return None
    if len(p) < 2:
        return None
    rows = get_articles(t, full_text, limit)
    if email:
        rows += get_filings(t, email)
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    rec = info.get("recommendationMean")  # 1 = strong buy ... 5 = sell
    return {
        "ticker": t,
        "name": info.get("shortName", t),
        "sent": mean(r["score"] for r in rows) if rows else 0.0,
        "mom": float(p.iloc[-1] / p.iloc[0] - 1),
        "upside": (target / price - 1) if price and target else 0.0,
        "rating": (3 - rec) if rec else 0.0,
        "rows": rows,
    }


def yahoo_screen(name, n=15):
    try:
        res = yf.screen(name, count=n)
        return [q["symbol"] for q in res.get("quotes", [])][:n]
    except Exception:
        return []


# ---------- Scoring ----------
def z(values):
    sd = pstdev(values) or 1
    m = mean(values)
    return [(v - m) / sd for v in values]


def rank(df, wn, wm, wa):
    df["z_news"] = z(df.sent.tolist())
    df["z_mom"] = z(df.mom.tolist())
    df["z_analyst"] = [(a + b) / 2 for a, b in zip(z(df.upside.tolist()), z(df.rating.tolist()))]
    df["score"] = (wn * df.z_news + wm * df.z_mom + wa * df.z_analyst) / ((wn + wm + wa) or 1)
    return df.sort_values("score", ascending=False)


LABELS = {"z_news": ("positive news", "negative news"),
          "z_mom": ("strong price trend", "weak price trend"),
          "z_analyst": ("analysts bullish", "analysts bearish")}


def why(r, good):
    comps = sorted(LABELS, key=lambda k: r[k], reverse=good)[:2]
    out = [LABELS[k][0 if good else 1] for k in comps if (r[k] > 0) == good]
    return ", ".join(out) or "mixed signals"


def card(r):
    c = "#2ecc8f" if r.score > thr else "#ff6b6b" if r.score < -thr else "#8b95a7"
    chips = why(r, r.score > 0).split(", ") + [f"3-mo {r.mom:+.0%}"]
    if r.upside:
        chips.append(f"analyst target {r.upside:+.0%}")
    chip_html = "".join(f"<span>{escape(x)}</span>" for x in chips)
    st.markdown(
        f'<div class="card" style="--c:{c}"><div class="top"><div>'
        f'<span class="tk">{escape(r.ticker)}</span><div class="nm">{escape(str(r["name"]))}</div></div>'
        f'<span class="pill">{r.score:+.2f}</span></div><div class="chips">{chip_html}</div></div>',
        unsafe_allow_html=True)


# ---------- Yahoo stock lists ----------
def _screen(q, n, **kw):
    for arg in ("size", "count"):
        try:
            res = yf.screen(q, **{arg: n}, **kw)
            return [x["symbol"] for x in res.get("quotes", [])][:n]
        except Exception:
            continue
    return []


@st.cache_data(ttl=900, show_spinner=False)
def universe(kind, n):
    EQ = yf.EquityQuery
    syms = []
    try:
        if kind == "big":
            q = EQ("and", [EQ("eq", ["region", "us"]), EQ("gt", ["intradaymarketcap", 50e9])])
            syms = _screen(q, n, sortField="intradaymarketcap", sortAsc=False)
        else:
            q = EQ("and", [EQ("eq", ["region", "us"]), EQ("btwn", ["intradayprice", 1, 10]),
                           EQ("gt", ["intradaymarketcap", 100e6]), EQ("gt", ["dayvolume", 500000])])
            syms = _screen(q, n, sortField="dayvolume", sortAsc=False)
    except Exception:
        pass
    return syms or _screen("undervalued_large_caps" if kind == "big" else "small_cap_gainers", n)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    scan_n = st.slider("Stocks to scan per tab", 5, 30, 15)
    days = st.slider("Price lookback (days)", 30, 365, 90)
    limit = st.slider("Articles per stock", 3, 15, 6)
    full_text = st.checkbox("Read full article text (slower)", False)
    email = st.text_input("Email for SEC filings (optional)")
    st.subheader("What matters most")
    wn = st.slider("News sentiment", 0.0, 3.0, 1.0, 0.1)
    wm = st.slider("Price trend", 0.0, 3.0, 1.0, 0.1)
    wa = st.slider("Analyst opinion", 0.0, 3.0, 1.0, 0.1)
    thr = st.slider("How picky (higher = fewer picks)", 0.2, 1.5, 0.5, 0.05)

st.markdown('<div class="hero"><h1>📈 Stock Picker</h1>'
            '<p>Scans stocks from Yahoo and ranks them using news, price trend and analyst '
            "opinion. It's a ranking, not a prediction, and not financial advice.</p></div>",
            unsafe_allow_html=True)


def scan(key, tickers):
    results, bar = [], st.progress(0.0)
    for i, t in enumerate(tickers):
        bar.progress((i + 1) / len(tickers), text=f"Analyzing {t} ({i + 1}/{len(tickers)})...")
        r = analyze(t, full_text, limit, email.strip(), days)
        if r:
            results.append(r)
    bar.empty()
    st.session_state[key] = results


def render(key):
    results = st.session_state.get(key)
    if not results:
        return
    if len(results) < 3:
        st.warning("Not enough stocks with data to compare. Try scanning again.")
        return
    df = rank(pd.DataFrame(results), wn, wm, wa)
    buys, avoids = df[df.score > thr], df[df.score < -thr]
    holds = df[(df.score <= thr) & (df.score >= -thr)]

    if len(buys):
        top = buys.iloc[0]
        st.markdown(
            f'<div class="pick"><small>Top pick</small><div class="t">{escape(top.ticker)}</div>'
            f'<div class="n">{escape(str(top["name"]))} · {why(top, True)}</div></div>',
            unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Buy", len(buys))
    m2.metric("⚪ Hold", len(holds))
    m3.metric("🚫 Don't buy", len(avoids))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Buy")
        for _, r in buys.iterrows():
            card(r)
        if buys.empty:
            st.info("Nothing stands out enough right now.")
    with c2:
        st.subheader("🚫 Don't buy")
        for _, r in avoids.iterrows():
            card(r)
        if avoids.empty:
            st.info("Nothing looks clearly bad right now.")

    if len(holds):
        with st.expander(f"⚪ Hold / no strong signal ({len(holds)})"):
            for _, r in holds.iterrows():
                card(r)

    st.subheader("Full ranking")
    table = df[["ticker", "name", "score", "sent", "mom", "upside"]].rename(columns={
        "sent": "news sentiment", "mom": "price change", "upside": "analyst upside"})
    st.dataframe(table.style.format({"score": "{:+.2f}", "news sentiment": "{:+.2f}",
                                     "price change": "{:+.1%}", "analyst upside": "{:+.1%}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("What's behind a score")
    pick = st.selectbox("Stock", df.ticker.tolist(), key=f"sel_{key}")
    for row in sorted(df[df.ticker == pick].iloc[0]["rows"], key=lambda x: x["score"]):
        icon = "🟢" if row["score"] > 0.05 else "🔴" if row["score"] < -0.05 else "⚪"
        st.markdown(f"{icon} `{row['score']:+.2f}` **{row['type']}**: [{row['title']}]({row['url']})")


tab_big, tab_cheap, tab_mine = st.tabs(["🏢 Big stocks", "🌱 Cheap stocks", "📝 My list"])

with tab_big:
    st.caption("Scans the largest US companies on Yahoo (over $50 billion).")
    if st.button("Scan big stocks", type="primary", use_container_width=True, key="go_big"):
        with st.spinner("Getting the list from Yahoo..."):
            syms = universe("big", scan_n)
        if len(syms) < 3:
            st.error("Couldn't get the list from Yahoo. Try again, or use the My list tab.")
        else:
            scan("big", syms)
    render("big")

with tab_cheap:
    st.caption("Scans active US stocks priced $1 to $10 with a market value over $100 million.")
    st.warning("Cheap stocks can swing wildly and some are very risky. "
               "A good score here is still not a safe bet.")
    if st.button("Scan cheap stocks", type="primary", use_container_width=True, key="go_cheap"):
        with st.spinner("Getting the list from Yahoo..."):
            syms = universe("cheap", scan_n)
        if len(syms) < 3:
            st.error("Couldn't get the list from Yahoo. Try again, or use the My list tab.")
        else:
            scan("cheap", syms)
    render("cheap")

with tab_mine:
    watch = st.text_area("Your tickers (comma-separated, max 30)",
                         "NVDA, AMD, INTC, QCOM, AAPL, MSFT, TSLA, AMZN", key="watch")
    if st.button("Scan my list", type="primary", use_container_width=True, key="go_mine"):
        tickers = list(dict.fromkeys(t for t in re.split(r"[,\s]+", watch.upper()) if t))[:30]
        if len(tickers) < 3:
            st.error("Enter at least 3 tickers so they can be compared.")
        else:
            scan("mine", tickers)
    render("mine")
