"""
agents.py - Enhanced Agentic AI System v2
==========================================
- Structured BUY/SELL/HOLD responses with confidence scores
- Smart PostgreSQL memory across sessions
- Email alerts for RSI/MACD/Bollinger breakouts
"""

import os, json, requests, datetime, numpy as np, smtplib, threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
from db import get_connection, get_cursor

client         = Groq(api_key=os.environ.get("GROQ_API_KEY"))
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
POLYGON_KEY    = os.environ.get("POLYGON_API_KEY", "")
MODEL          = "llama-3.3-70b-versatile"
EMAIL_SENDER   = os.environ.get("ALERT_EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("ALERT_EMAIL_RECEIVER")

COMPANY_NAMES = {
    'AAPL':'Apple','NVDA':'NVIDIA','MSFT':'Microsoft','GOOGL':'Alphabet Google',
    'META':'Meta Platforms','AMZN':'Amazon','TSLA':'Tesla','AMD':'AMD',
    'INTC':'Intel','JPM':'JPMorgan','BAC':'Bank of America','V':'Visa',
    'MA':'Mastercard','NFLX':'Netflix','DIS':'Disney','COIN':'Coinbase',
    'PYPL':'PayPal','UBER':'Uber','SHOP':'Shopify','PLTR':'Palantir',
}

# Index ETFs - their price is NOT the same as the index level they track.
# The app has no live feed for index levels, so any index number must not be fabricated.
INDEX_ETFS = {
    "SPY": ("S&P 500 index", "the SPDR S&P 500 ETF"),
    "VOO": ("S&P 500 index", "the Vanguard S&P 500 ETF"),
    "QQQ": ("Nasdaq-100 index", "the Invesco QQQ ETF"),
    "DIA": ("Dow Jones index", "the SPDR Dow Jones ETF"),
    "IWM": ("Russell 2000 index", "the iShares Russell 2000 ETF"),
}



# - MEMORY -

def save_message(user_id, role, message, agent_used=None):
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO chat_history (user_id,role,message,agent_used) VALUES (%s,%s,%s,%s)",
                    (user_id, role, message, agent_used))
        conn.commit()
    finally:
        cur.close(); conn.close()


def get_chat_history(user_id, limit=10):
    conn = get_connection(); cur = get_cursor(conn)
    try:
        cur.execute("SELECT role,message,agent_used,created_at FROM chat_history WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit))
        return list(reversed([dict(r) for r in cur.fetchall()]))
    finally:
        cur.close(); conn.close()


def get_user_portfolio(user_id):
    conn = get_connection(); cur = get_cursor(conn)
    try:
        cur.execute("SELECT symbol,quantity,avg_cost FROM portfolio_holdings WHERE user_id=%s", (user_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def get_user_preferences(user_id):
    conn = get_connection(); cur = get_cursor(conn)
    try:
        cur.execute("SELECT preferences,name,email FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        return {"name": row["name"], "email": row.get("email"), **(row["preferences"] or {})} if row else {}
    finally:
        cur.close(); conn.close()


def save_agent_memory(user_id, content, memory_type="fact"):
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM agent_memory WHERE user_id=%s AND content=%s", (user_id, content))
        if not cur.fetchone():
            cur.execute("INSERT INTO agent_memory (user_id,memory_type,content) VALUES (%s,%s,%s)",
                        (user_id, memory_type, content))
            conn.commit()
    finally:
        cur.close(); conn.close()


def get_agent_memories(user_id, limit=8):
    conn = get_connection(); cur = get_cursor(conn)
    try:
        cur.execute("SELECT content,memory_type FROM agent_memory WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def extract_and_save_preferences(user_id, message):
    keywords = {
        "preference": ["prefer","like","love","favorite","focus on","interested in"],
        "avoid":      ["avoid","dont like","hate","not interested","stay away"],
        "risk":       ["risk","aggressive","conservative","moderate","safe"],
        "goal":       ["goal","target","want to","trying to","planning to"],
        "watchlist":  ["watching","monitor","track","keep eye on"],
    }
    msg_lower = message.lower()
    for mem_type, words in keywords.items():
        if any(w in msg_lower for w in words):
            save_agent_memory(user_id, message, memory_type=mem_type)
            break


# - DATA & INDICATORS -

def fetch_stock_quote(symbol):
    """Fetch live quote - Finnhub first, Polygon fallback, yfinance last."""
    sym = symbol.upper()

    # 1. Finnhub - 60 req/min free
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if finnhub_key:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/quote?symbol={sym}&token={finnhub_key}",
                timeout=10).json()
            price = float(r.get("c", 0) or 0)
            prev  = float(r.get("pc", 0) or 0)
            if price > 0:
                return {"symbol": sym, "price": round(price, 2),
                        "prev_close": round(prev, 2),
                        "change_pct": round(((price - prev) / prev * 100), 2) if prev else 0,
                        "volume": 0}
        except Exception:
            pass

    # 2. Polygon fallback
    if POLYGON_KEY:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?apiKey={POLYGON_KEY}"
            r   = requests.get(url, timeout=10).json()
            res = r.get("results", [{}])[0] if r.get("results") else {}
            price = float(res.get("c", 0) or 0)
            prev  = float(res.get("o", price) or price)
            if price > 0:
                return {"symbol": sym, "price": round(price, 2),
                        "prev_close": round(prev, 2),
                        "change_pct": round(((price - prev) / prev * 100), 2) if prev else 0,
                        "volume": int(res.get("v", 0))}
        except Exception:
            pass

    # 3. yfinance last resort
    try:
        import yfinance as yf
        info  = yf.Ticker(sym).fast_info
        price = float(info.last_price or 0)
        prev  = float(info.previous_close or 0)
        if price > 0:
            return {"symbol": sym, "price": round(price, 2),
                    "prev_close": round(prev, 2),
                    "change_pct": round(((price - prev) / prev * 100), 2) if prev else 0,
                    "volume": int(info.three_month_average_volume or 0)}
    except Exception:
        pass

    return {"symbol": sym, "price": 0, "prev_close": 0, "change_pct": 0}


def fetch_price_history(symbol, period="3mo"):
    """Fetch price history - Twelve Data first, Polygon fallback, yfinance last."""
    sym = symbol.upper()

    # 1. Twelve Data - 800 req/day, no per-minute limit issues
    twelve_key = os.environ.get("TWELVE_DATA_API_KEY", "")
    if twelve_key:
        try:
            period_map = {'5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            outputsize = period_map.get(period, 90)
            url = (f"https://api.twelvedata.com/time_series?symbol={sym}"
                   f"&interval=1day&outputsize={outputsize}&apikey={twelve_key}")
            r = requests.get(url, timeout=15).json()
            values = r.get("values", [])
            if values:
                data = [{"date":   v["datetime"],
                         "close":  round(float(v["close"]), 2),
                         "high":   round(float(v["high"]), 2),
                         "low":    round(float(v["low"]), 2),
                         "volume": int(float(v.get("volume", 0)))}
                        for v in reversed(values)]
                return data
        except Exception:
            pass

    # 2. Polygon fallback
    if POLYGON_KEY:
        try:
            from datetime import date, timedelta
            period_days = {'5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            days  = period_days.get(period, 90)
            end   = date.today()
            start = end - timedelta(days=days)
            url   = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
                     f"{start}/{end}?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_KEY}")
            r = requests.get(url, timeout=15).json()
            results = r.get("results", [])
            if results:
                return [{"date": datetime.datetime.fromtimestamp(d["t"]/1000).strftime("%Y-%m-%d"),
                         "close": round(d.get("c", 0), 2),
                         "high":  round(d.get("h", 0), 2),
                         "low":   round(d.get("l", 0), 2),
                         "volume": int(d.get("v", 0))} for d in results]
        except Exception:
            pass

    # 3. yfinance last resort
    try:
        import yfinance as yf
        hist = yf.Ticker(sym).history(period=period, interval="1d")
        if hist.empty:
            return []
        return [{"date": str(d)[:10], "close": round(float(r["Close"]), 2),
                 "high": round(float(r["High"]), 2), "low": round(float(r["Low"]), 2),
                 "volume": int(r["Volume"])} for d, r in hist.iterrows()]
    except Exception:
        return []


def fetch_stock_news(symbol):
    """Fetch stock news using NewsAPI."""
    try:
        if NEWS_API_KEY:
            query = COMPANY_NAMES.get(symbol.upper(), symbol)
            url   = f"https://newsapi.org/v2/everything?q={query}+stock&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}&language=en"
            r     = requests.get(url, timeout=10).json()
            return [{"title": a.get("title", ""), "source": a.get("source", {}).get("name", ""),
                     "sentiment": "Neutral", "score": 0.0, "url": a.get("url", "#")}
                    for a in r.get("articles", [])[:5]
                    if a.get("title") and "[Removed]" not in a.get("title", "")]
        return []
    except Exception:
        return []


def compute_indicators(prices):
    if len(prices) < 26:
        return {}
    closes = np.array([p["close"] for p in prices])

    # RSI
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag, al = np.mean(gains[:14]), np.mean(losses[:14])
    for i in range(14, len(gains)):
        ag = (ag * 13 + gains[i]) / 14
        al = (al * 13 + losses[i]) / 14
    rsi = round(100 - (100 / (1 + ag / al)) if al else 100, 2)

    # MACD
    def ema(d, p):
        k, e = 2 / (p + 1), d[0]
        v = [e]
        for x in d[1:]:
            e = x * k + e * (1 - k)
            v.append(e)
        return np.array(v)

    ema12     = ema(closes, 12)
    ema26     = ema(closes, 26)
    macd_line = ema12 - ema26
    signal    = ema(macd_line, 9)
    histogram = float(macd_line[-1] - signal[-1])

    # Bollinger Bands
    recent   = closes[-20:]
    bb_mid   = np.mean(recent)
    bb_std   = np.std(recent)
    bb_upper = round(float(bb_mid + 2 * bb_std), 2)
    bb_lower = round(float(bb_mid - 2 * bb_std), 2)
    bb_mid   = round(float(bb_mid), 2)
    price    = float(closes[-1])

    signals = []
    if rsi > 70:
        signals.append("RSI overbought")
    elif rsi < 30:
        signals.append("RSI oversold")
    if histogram > 0:
        signals.append("MACD bullish")
    else:
        signals.append("MACD bearish")
    if price > bb_upper:
        signals.append("Above upper Bollinger Band")
    elif price < bb_lower:
        signals.append("Below lower Bollinger Band")

    bullish    = sum(1 for s in signals if any(w in s for w in ["oversold", "bullish", "Below Bollinger"]))
    bearish    = sum(1 for s in signals if any(w in s for w in ["overbought", "bearish", "Above Bollinger"]))
    overall    = "BUY" if bullish > bearish else "SELL" if bearish > bullish else "HOLD"
    confidence = min(95, 50 + max(bullish, bearish) * 15)

    return {"rsi": rsi, "macd": round(float(macd_line[-1]), 4),
            "signal_line": round(float(signal[-1]), 4), "histogram": round(histogram, 4),
            "bb_upper": bb_upper, "bb_middle": bb_mid, "bb_lower": bb_lower,
            "price": round(price, 2), "signals": signals, "overall": overall, "confidence": confidence}


# - EMAIL -

def send_email_alert(subject, body_html, receiver=None):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return False
    to = receiver or EMAIL_RECEIVER
    if not to:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = to
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, to, msg.as_string())
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


def format_alert_email(alerts, user_name="Investor"):
    rows = ""
    for a in alerts:
        c = "#4CAF8A" if a["type"] == "BUY" else "#C75B6A" if a["type"] == "SELL" else "#C4922A"
        rows += f"""<tr>
            <td style="padding:10px;border-bottom:1px solid #1A2744;font-family:monospace;color:#D6E0F5;font-weight:bold">{a["symbol"]}</td>
            <td style="padding:10px;border-bottom:1px solid #1A2744;color:{c};font-weight:bold">{a["type"]}</td>
            <td style="padding:10px;border-bottom:1px solid #1A2744;color:#D6E0F5">${a["price"]}</td>
            <td style="padding:10px;border-bottom:1px solid #1A2744;color:#6B82AA">{a["reason"]}</td>
        </tr>"""
    return f"""<html><body style="background:#0A0F1E;color:#D6E0F5;font-family:sans-serif;padding:30px">
<h2 style="color:#5B8DEF;font-family:monospace">MARKET INTELLIGENCE ALERTS</h2>
<p style="color:#6B82AA">Hi {user_name}, alerts as of {datetime.datetime.now().strftime("%b %d %H:%M")}:</p>
<table style="width:100%;border-collapse:collapse;background:#0F1629;margin:20px 0">
<tr style="background:#141D35">
<th style="padding:10px;text-align:left;color:#6B82AA;font-size:11px">SYMBOL</th>
<th style="padding:10px;text-align:left;color:#6B82AA;font-size:11px">SIGNAL</th>
<th style="padding:10px;text-align:left;color:#6B82AA;font-size:11px">PRICE</th>
<th style="padding:10px;text-align:left;color:#6B82AA;font-size:11px">REASON</th>
</tr>{rows}</table>
<p style="color:#2E3F62;font-size:11px">Not financial advice.</p>
</body></html>"""


# - SPECIALIST AGENTS -

def technical_agent(symbol, question):
    prices = fetch_price_history(symbol)
    if not prices:
        return f"Could not fetch price data for {symbol}."
    ind   = compute_indicators(prices)
    quote = fetch_stock_quote(symbol)
    context = f"""Symbol: {symbol} | Price: ${ind.get("price", quote.get("price"))} | Change: {quote.get("change_pct", 0):+.2f}%
RSI: {ind.get("rsi")} | MACD: {ind.get("macd")} | Signal: {ind.get("signal_line")} | Histogram: {ind.get("histogram")}
BB Upper: ${ind.get("bb_upper")} | Middle: ${ind.get("bb_middle")} | Lower: ${ind.get("bb_lower")}
Signals: {", ".join(ind.get("signals", []))}
Overall: {ind.get("overall")} ({ind.get("confidence")}% confidence)
Question: {question}"""
    messages = [
        {"role": "system", "content": f"""You are an expert technical analyst.
Format your response EXACTLY like this:

## SIGNAL: {ind.get("overall","HOLD")} ({ind.get("confidence",50)}% confidence)
**Price:** $X | **Timeframe:** Short-term (1-2 weeks)

### Indicator Summary
- RSI ({ind.get("rsi")}): [interpretation]
- MACD ({ind.get("macd")}): [interpretation]
- Bollinger Bands: [position and meaning]

### Key Insight
[2-3 sentences on the most important signal]

### Action
[Clear specific recommendation]

Not financial advice."""},
        {"role": "user", "content": context}
    ]
    return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=600).choices[0].message.content


def research_agent(symbol, question):
    quote  = fetch_stock_quote(symbol)
    news   = fetch_stock_news(symbol)
    prices = fetch_price_history(symbol, period="1mo")
    ind    = compute_indicators(prices) if prices else {}
    headlines = "\n".join([f"- {n.get('title', '')}" for n in news[:4]])
    context = f"""Symbol: {symbol} | Price: ${quote.get("price")} ({quote.get("change_pct", 0):+.2f}%)
RSI: {ind.get("rsi", "N/A")}
Headlines:
{headlines}
Question: {question}"""
    messages = [
        {"role": "system", "content": """Format EXACTLY:

## RESEARCH: {SYMBOL}
### Market Sentiment: [BULLISH/BEARISH/NEUTRAL]

### Key Developments
[3-4 bullet points]

### News Analysis
[2-3 sentences]

### Bottom Line
[1-2 sentences clear takeaway]

Not financial advice."""},
        {"role": "user", "content": context}
    ]
    return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=600).choices[0].message.content


def portfolio_agent(user_id, question):
    portfolio = get_user_portfolio(user_id)
    prefs     = get_user_preferences(user_id)
    memories  = get_agent_memories(user_id, limit=5)
    if not portfolio:
        return "No stocks in portfolio yet. Add stocks to get personalized analysis."
    holdings_data = []
    total_value   = 0
    for h in portfolio:
        quote   = fetch_stock_quote(h["symbol"])
        price   = quote.get("price", h["avg_cost"])
        curr    = price * h["quantity"]
        cost    = h["avg_cost"] * h["quantity"]
        pnl     = curr - cost
        pnl_pct = (pnl / cost * 100) if cost else 0
        total_value += curr
        holdings_data.append({"symbol": h["symbol"], "quantity": h["quantity"],
                               "avg_cost": h["avg_cost"], "current_price": price,
                               "value": round(curr, 2), "pnl": round(pnl, 2),
                               "pnl_pct": round(pnl_pct, 2)})
    for h in holdings_data:
        h["weight"] = round(h["value"] / total_value * 100, 1) if total_value else 0
    mem_ctx = "\n".join([f"- [{m['memory_type']}] {m['content']}" for m in memories])
    context = f"""User: {prefs.get("name")} | Risk: {prefs.get("riskAppetite")} | Goal: {prefs.get("investmentGoal")}
Preferences: {mem_ctx or "None recorded"}
Portfolio (Total: ${round(total_value, 2)}):
{json.dumps(holdings_data, indent=2)}
Question: {question}"""
    messages = [
        {"role": "system", "content": """Format EXACTLY:

## PORTFOLIO ANALYSIS
**Total Value:** $X

### Top Performers
[Top 2-3 by P&L%]

### Risk Flags
[Concentration, underperformers]

### Rebalancing Actions
1. [specific action]
2. [specific action]
3. [specific action]

Not financial advice."""},
        {"role": "user", "content": context}
    ]
    return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=700).choices[0].message.content


def sentiment_agent(symbol, question):
    news = fetch_stock_news(symbol)
    if not news:
        return f"Could not fetch news for {symbol}."
    headlines = "\n".join([f"- {n.get('title', '')}" for n in news[:5]])
    context = f"Symbol: {symbol} | Articles: {len(news)}\nHeadlines:\n{headlines}\nQuestion: {question}"
    messages = [
        {"role": "system", "content": """Format EXACTLY:

## SENTIMENT: {SYMBOL}

### Key Headlines
[summarize top 3 headlines]

### Bullish Themes
[Key positive narratives]

### Bearish Concerns
[Key risks]

### Trading Implication
[1 sentence on short-term price impact]

Not financial advice."""},
        {"role": "user", "content": context}
    ]
    return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=500).choices[0].message.content


def alert_agent(user_id, question):
    portfolio = get_user_portfolio(user_id)
    prefs     = get_user_preferences(user_id)
    if not portfolio:
        return "No portfolio found. Add stocks to enable alerts."
    alerts        = []
    alert_objects = []
    for h in portfolio:
        sym    = h["symbol"]
        prices = fetch_price_history(sym, "1mo")
        if not prices:
            continue
        ind   = compute_indicators(prices)
        quote = fetch_stock_quote(sym)
        price = quote.get("price", 0)
        rsi   = ind.get("rsi", 50)
        hist  = ind.get("histogram", 0)
        bb_u  = ind.get("bb_upper", 0)
        bb_l  = ind.get("bb_lower", 0)
        if rsi > 75:
            alerts.append(f"RED {sym} RSI={rsi} - Strongly overbought.")
            alert_objects.append({"symbol": sym, "type": "SELL", "price": price, "reason": f"RSI={rsi} strongly overbought"})
        elif rsi > 70:
            alerts.append(f"YELLOW {sym} RSI={rsi} - Overbought.")
            alert_objects.append({"symbol": sym, "type": "SELL", "price": price, "reason": f"RSI={rsi} overbought"})
        elif rsi < 25:
            alerts.append(f"GREEN {sym} RSI={rsi} - Strongly oversold.")
            alert_objects.append({"symbol": sym, "type": "BUY", "price": price, "reason": f"RSI={rsi} strongly oversold"})
        elif rsi < 30:
            alerts.append(f"GREEN {sym} RSI={rsi} - Oversold.")
            alert_objects.append({"symbol": sym, "type": "BUY", "price": price, "reason": f"RSI={rsi} oversold"})
        if hist > 0.5:
            alerts.append(f"UP {sym} MACD={hist:.3f} - Bullish momentum.")
            alert_objects.append({"symbol": sym, "type": "BUY", "price": price, "reason": f"MACD {hist:.3f} bullish"})
        elif hist < -0.5:
            alerts.append(f"DOWN {sym} MACD={hist:.3f} - Bearish momentum.")
            alert_objects.append({"symbol": sym, "type": "SELL", "price": price, "reason": f"MACD {hist:.3f} bearish"})
        if price and bb_u and price > bb_u * 1.02:
            alerts.append(f"WARNING {sym} ${price} above BB upper ${bb_u}.")
        elif price and bb_l and price < bb_l * 0.98:
            alerts.append(f"INFO {sym} ${price} below BB lower ${bb_l}.")
    if not alerts:
        return "All Clear - No critical alerts. All positions within normal technical ranges."
    if alert_objects and EMAIL_SENDER:
        html = format_alert_email(alert_objects, prefs.get("name", "Investor"))
        threading.Thread(target=send_email_alert,
                         args=(f"Market Intelligence: {len(alert_objects)} Alert(s)", html, prefs.get("email")),
                         daemon=True).start()
    messages = [
        {"role": "system", "content": f"""Format EXACTLY:

## PORTFOLIO ALERTS ({len(alerts)} found)

[list alerts clearly]

### Priority Actions
1. [most urgent]
2. [second priority]
3. [third]

Not financial advice."""},
        {"role": "user", "content": "Alerts:\n" + "\n".join(alerts) + f"\nQuestion: {question}"}
    ]
    return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=600).choices[0].message.content




def portfolio_intel_agent(user_id, new_symbol, removed_symbol=None):
    """Analyze portfolio risk after adding or removing a stock."""
    # Resolve email to UUID
    if user_id and "@" in str(user_id):
        conn = get_connection(); cur = get_cursor(conn)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (user_id,))
            row = cur.fetchone()
            if row:
                user_id = str(row["id"])
        finally:
            cur.close(); conn.close()

    portfolio = get_user_portfolio(user_id)
    prefs     = get_user_preferences(user_id)
    if not portfolio:
        return {"error": "No portfolio found"}

    holdings_analysis = []
    total_value = 0
    total_cost  = 0
    risk_scores = []

    for h in portfolio:
        sym    = h["symbol"]
        prices = fetch_price_history(sym, "3mo")
        quote  = fetch_stock_quote(sym)
        price  = quote.get("price", h["avg_cost"])
        ind    = compute_indicators(prices) if len(prices) >= 26 else {}
        curr   = price * h["quantity"]
        cost   = h["avg_cost"] * h["quantity"]
        pnl    = curr - cost
        pnl_pct = (pnl / cost * 100) if cost else 0
        total_value += curr
        total_cost  += cost
        rsi  = ind.get("rsi", 50)
        hist = ind.get("histogram", 0)
        risk = 50
        if rsi > 75:       risk += 25
        elif rsi > 70:     risk += 15
        elif rsi < 30:     risk -= 15
        if hist < -0.5:    risk += 10
        elif hist > 0.5:   risk -= 10
        if pnl_pct < -15:  risk += 15
        elif pnl_pct > 30: risk += 10
        risk = max(0, min(100, risk))
        risk_scores.append(risk)
        signal = ind.get("overall", "HOLD")
        trend  = "UPTREND" if hist > 0 and rsi < 70 else "DOWNTREND" if hist < 0 and rsi > 30 else "NEUTRAL"
        holdings_analysis.append({
            "symbol": sym, "price": price, "pnl_pct": round(pnl_pct, 2),
            "rsi": round(rsi, 1) if rsi else "N/A", "signal": signal,
            "trend": trend, "risk": risk, "is_new": sym == new_symbol.upper(),
        })

    portfolio_pnl = ((total_value - total_cost) / total_cost * 100) if total_cost else 0
    avg_risk      = round(sum(risk_scores) / len(risk_scores)) if risk_scores else 50
    risk_label    = "HIGH" if avg_risk > 65 else "MODERATE" if avg_risk > 40 else "LOW"

    lines = [
        f"- {h['symbol']}: Price=${h['price']}, P&L={h['pnl_pct']:+.1f}%,"
        f" RSI={h['rsi']}, Signal={h['signal']}, Trend={h['trend']}, Risk={h['risk']}/100"
        for h in holdings_analysis
    ]
    holdings_str = "\n".join(lines)

    if removed_symbol:
        action       = f"removed {removed_symbol} from"
        action_label = f"Impact of Removing {removed_symbol}"
    else:
        action       = f"added {new_symbol} to"
        action_label = f"Newly Added: {new_symbol}"

    context = (
        f"User just {action} their portfolio.\n"
        f"User profile: Risk={prefs.get('riskAppetite','Unknown')}, Goal={prefs.get('investmentGoal','Unknown')}\n\n"
        f"Current Portfolio (Total=${round(total_value,2)}, P&L={portfolio_pnl:+.1f}%):\n"
        f"{holdings_str}\n\n"
        f"Portfolio Risk Score: {avg_risk}/100 ({risk_label} RISK)"
    )

    system_prompt = (
        f"You are a portfolio intelligence AI. The user just {action} their portfolio.\n"
        "Give a sharp, actionable analysis in this EXACT format:\n\n"
        "## PORTFOLIO INTELLIGENCE REPORT\n\n"
        "### Overall Risk: [LOW/MODERATE/HIGH] ([score]/100)\n"
        "[1 sentence on overall portfolio health]\n\n"
        f"### {action_label}\n"
        "[2-3 sentences: impact of this change, what to watch]\n\n"
        "### Portfolio Trends\n"
        "[One bullet per stock with strong signal: symbol, direction, reason]\n\n"
        "### Sell Recommendations\n"
        "[1-3 specific stocks with sell conditions]\n\n"
        "### Best Timing Window\n"
        "[1-2 sentences on ideal timing to rebalance]\n\n"
        "Keep it concise and actionable. Not financial advice."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": context},
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=700).choices[0].message.content

    return {
        "response":      response,
        "risk_score":    avg_risk,
        "risk_label":    risk_label,
        "portfolio_pnl": round(portfolio_pnl, 2),
        "total_value":   round(total_value, 2),
        "holdings":      holdings_analysis,
    }


# - ORCHESTRATOR -

def detect_intent(message):
    prompt = f"""Return ONLY valid JSON. Classify this financial question.

Question: "{message}"

Rules:
- "technical" = requests for technical analysis, chart analysis, indicators, buy/sell signals, RSI/MACD
- "research" = requests for research, news, fundamentals, why is X moving
- "portfolio" = questions about the user's portfolio, holdings, P&L
- "sentiment" = questions about market sentiment, news sentiment
- "alert" = requests to check alerts, warnings
- "general" = simple price questions ("what is X price", "what is closing rate"), greetings, general chat, comparisons, recommendations

Fields: "agent" (technical/research/portfolio/sentiment/alert/general), "symbol" (stock ticker or null)
Examples:
{{"agent":"technical","symbol":"AAPL"}} for "analyze AAPL" or "should I buy AAPL"
{{"agent":"general","symbol":"AAPL"}} for "what is AAPL price" or "closing rate of apple"
{{"agent":"general","symbol":null}} for "hello" or "what stocks should I watch"
{{"agent":"portfolio","symbol":null}} for "how is my portfolio doing"

Return ONLY the JSON, nothing else."""
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=80)
    try:
        text = resp.choices[0].message.content.strip()
        return json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return {"agent": "general", "symbol": None}


def orchestrate(user_id, message):
    save_message(user_id, "user", message)
    memories = get_agent_memories(user_id, 5)
    prefs    = get_user_preferences(user_id)
    extract_and_save_preferences(user_id, message)
    intent = detect_intent(message)
    agent  = intent.get("agent", "general")
    symbol = intent.get("symbol")
    if agent == "technical" and symbol:
        response_text = technical_agent(symbol, message)
    elif agent == "research" and symbol:
        response_text = research_agent(symbol, message)
    elif agent == "portfolio":
        response_text = portfolio_agent(user_id, message)
    elif agent == "sentiment" and symbol:
        response_text = sentiment_agent(symbol, message)
    elif agent == "alert":
        response_text = alert_agent(user_id, message)
    else:
        history = get_chat_history(user_id, 6)
        mem_ctx = "\n".join([f"- [{m['memory_type']}] {m['content']}" for m in memories])
        # If a symbol is mentioned, fetch live price for context
        price_context = ""
        if symbol:
            try:
                quote = fetch_stock_quote(symbol)
                if quote.get("price", 0) > 0:
                    price_context = f"\nLive data for {symbol}: Price=${quote['price']}, Change={quote['change_pct']:+.2f}%, Prev Close=${quote['prev_close']}"
                    # Flag index/ETF distinction so the agent never passes off an ETF price as the index value
                    if symbol.upper() in INDEX_ETFS:
                        index_name, etf_desc = INDEX_ETFS[symbol.upper()]
                        price_context += (f"\nIMPORTANT: This price is for {symbol.upper()}, which is {etf_desc}. "
                                          f"It is NOT the same number as {index_name}. If the user asked about "
                                          f"{index_name}, clearly tell them this is the ETF price and that the index "
                                          f"level is a different (much larger) number you cannot fetch directly.")
            except Exception:
                pass
        system  = f"""You are Market Intelligence - a friendly, conversational AI stock advisor.
User: {prefs.get("name","Investor")} | Risk: {prefs.get("riskAppetite","Unknown")} | Goal: {prefs.get("investmentGoal","Unknown")}
Known preferences: {mem_ctx or "None yet - learn from this conversation."}
{price_context}
Be conversational and concise. Answer the specific question asked. For simple price questions just give the price directly.
For complex questions, be insightful. End investment advice with: Not financial advice."""
        msgs = [{"role": "system", "content": system}] + \
               [{"role": h["role"], "content": h["message"]} for h in history[-4:]] + \
               [{"role": "user", "content": message}]
        response_text = client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=300).choices[0].message.content
        # Deterministic safeguard for index ETFs.
        # Root cause: the app has the ETF price but NO source for the actual index level.
        # So any index number the LLM volunteers is fabricated (from memory/rule-of-thumb).
        # Fix: when the symbol is an index ETF, replace the entire response with a sourced,
        # honest answer that states only what we can actually verify - the ETF price - and
        # explicitly declines to give an index level we cannot fetch.
        if symbol and symbol.upper() in INDEX_ETFS:
            index_name, etf_desc = INDEX_ETFS[symbol.upper()]
            try:
                etf_quote = fetch_stock_quote(symbol)
                etf_price = etf_quote.get("price", 0)
                etf_chg   = etf_quote.get("change_pct", 0)
            except Exception:
                etf_price = 0
                etf_chg   = 0
            if etf_price > 0:
                response_text = (
                    f"I can show you {symbol.upper()} ({etf_desc}): "
                    f"${etf_price} ({etf_chg:+.2f}% today).\n\n"
                    f"Note: {symbol.upper()} tracks the {index_name} but is a different instrument - "
                    f"its price is not the index level. I don't have a live feed for the {index_name} "
                    f"value itself, so I won't estimate it (a rule-of-thumb multiplier would be a guess, "
                    f"not real data). For the exact index level, check a source like Google Finance or your broker.\n\n"
                    f"Not financial advice."
                )
            else:
                response_text = (
                    f"I track {symbol.upper()} ({etf_desc}), which follows the {index_name}, "
                    f"but I don't currently have a live price for it, and I don't have a direct feed "
                    f"for the {index_name} index level itself. Please check a market data source for the current value.\n\n"
                    f"Not financial advice."
                )
        agent = "general"

    # CATCH-ALL deterministic guard: no matter which agent ran, if the symbol is an
    # index ETF, ensure we never pass off the ETF price as the index level or let a
    # fabricated index number through. This fires even if detect_intent routed the
    # query to technical/research instead of general.
    if symbol and symbol.upper() in INDEX_ETFS:
        index_name, etf_desc = INDEX_ETFS[symbol.upper()]
        # If the response doesn't already make the ETF-vs-index distinction clear, override it.
        rt_lower = response_text.lower()
        makes_distinction = ("etf" in rt_lower and "not" in rt_lower and "index" in rt_lower)
        if not makes_distinction:
            try:
                etf_quote = fetch_stock_quote(symbol)
                etf_price = etf_quote.get("price", 0)
                etf_chg   = etf_quote.get("change_pct", 0)
            except Exception:
                etf_price = 0; etf_chg = 0
            if etf_price > 0:
                response_text = (
                    f"I can show you {symbol.upper()} ({etf_desc}): ${etf_price} ({etf_chg:+.2f}% today). "
                    f"It tracks the {index_name} but is a different instrument - its price is not the index level. "
                    f"I don't have a live feed for the {index_name} value itself, so I won't estimate it. "
                    f"For the exact index level, check a market data source.\n\nNot financial advice."
                )

    save_message(user_id, "assistant", response_text, agent)
    return {"response": response_text, "agent_used": agent, "symbol": symbol, "intent": intent}