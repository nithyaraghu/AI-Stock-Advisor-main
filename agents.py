"""
agents.py — Agentic AI system for AI Stock Advisor
=====================================================
5 specialist agents orchestrated by a master agent:
1. Research Agent    — news, earnings, market overview
2. Technical Agent   — RSI, MACD, Bollinger Bands
3. Portfolio Agent   — portfolio risk, rebalancing
4. Sentiment Agent   — news sentiment scoring
5. Alert Agent       — price threshold monitoring

All powered by Groq (free) + LangChain + PostgreSQL memory
"""

import os
import json
import requests
import datetime
import numpy as np
from groq import Groq
from db import get_connection, get_cursor
from config import settings

# ── Groq client ───────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "9ZQUXAH9JOQRSQDV")
MODEL = "llama-3.3-70b-versatile"   # best free model on Groq


# ══════════════════════════════════════════════════════════════
# MEMORY HELPERS — PostgreSQL chat history + agent memory
# ══════════════════════════════════════════════════════════════

def save_message(user_id: str, role: str, message: str, agent_used: str = None):
    """Save a chat message to PostgreSQL chat_history table."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, role, message, agent_used)
            VALUES (%s, %s, %s, %s)
        """, (user_id, role, message, agent_used))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_chat_history(user_id: str, limit: int = 10) -> list:
    """Retrieve last N messages for a user from PostgreSQL."""
    conn = get_connection()
    cur = get_cursor(conn)
    try:
        cur.execute("""
            SELECT role, message, agent_used, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        rows = cur.fetchall()
        # Reverse so oldest is first
        return list(reversed([dict(r) for r in rows]))
    finally:
        cur.close()
        conn.close()


def get_user_portfolio(user_id: str) -> list:
    """Get user's portfolio from PostgreSQL."""
    conn = get_connection()
    cur = get_cursor(conn)
    try:
        cur.execute("""
            SELECT symbol, quantity, avg_cost
            FROM portfolio_holdings
            WHERE user_id = %s
        """, (user_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_user_preferences(user_id: str) -> dict:
    """Get user preferences from PostgreSQL."""
    conn = get_connection()
    cur = get_cursor(conn)
    try:
        cur.execute(
            "SELECT preferences, name FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        if row:
            return {"name": row["name"], **row["preferences"]}
        return {}
    finally:
        cur.close()
        conn.close()


def save_agent_memory(user_id: str, content: str, memory_type: str = "fact"):
    """Save important facts about user to agent_memory table."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO agent_memory (user_id, memory_type, content)
            VALUES (%s, %s, %s)
        """, (user_id, memory_type, content))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_agent_memories(user_id: str, limit: int = 5) -> list:
    """Get recent agent memories for a user."""
    conn = get_connection()
    cur = get_cursor(conn)
    try:
        cur.execute("""
            SELECT content, memory_type, created_at
            FROM agent_memory
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════
# TOOL FUNCTIONS — what agents call to get real data
# ══════════════════════════════════════════════════════════════

def fetch_stock_quote(symbol: str) -> dict:
    """Fetch live stock quote from Alpha Vantage."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": ALPHA_VANTAGE_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json().get("Global Quote", {})
        return {
            "symbol":     data.get("01. symbol", symbol),
            "price":      float(data.get("05. price", 0)),
            "change":     float(data.get("09. change", 0)),
            "change_pct": data.get("10. change percent", "0%"),
            "volume":     data.get("06. volume", "N/A"),
            "prev_close": float(data.get("08. previous close", 0)),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def fetch_stock_news(symbol: str) -> list:
    """Fetch recent news + sentiment for a stock."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers":  symbol.upper(),
            "limit":    5,
            "apikey":   ALPHA_VANTAGE_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        articles = resp.json().get("feed", [])
        return [
            {
                "title":     a.get("title"),
                "source":    a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score":     a.get("overall_sentiment_score"),
                "time":      a.get("time_published"),
            }
            for a in articles[:5]
        ]
    except Exception as e:
        return [{"error": str(e)}]


def fetch_time_series(symbol: str, days: int = 30) -> list:
    """Fetch daily price data for technical analysis."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function":   "TIME_SERIES_DAILY",
            "symbol":     symbol.upper(),
            "outputsize": "compact",
            "apikey":     ALPHA_VANTAGE_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        ts = resp.json().get("Time Series (Daily)", {})
        prices = []
        for date, values in list(ts.items())[:days]:
            prices.append({
                "date":   date,
                "open":   float(values["1. open"]),
                "high":   float(values["2. high"]),
                "low":    float(values["3. low"]),
                "close":  float(values["4. close"]),
                "volume": int(values["5. volume"]),
            })
        return sorted(prices, key=lambda x: x["date"])
    except Exception as e:
        return [{"error": str(e)}]


def compute_rsi(prices: list, period: int = 14) -> float:
    """Calculate RSI from price data."""
    if len(prices) < period + 1:
        return None
    closes = [p["close"] for p in prices]
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi), 2)


def compute_macd(prices: list) -> dict:
    """Calculate MACD (12, 26, 9) from price data."""
    if len(prices) < 26:
        return {"macd": None, "signal": None, "histogram": None}
    closes = np.array([p["close"] for p in prices])

    def ema(data, period):
        k = 2 / (period + 1)
        ema_vals = [data[0]]
        for price in data[1:]:
            ema_vals.append(price * k + ema_vals[-1] * (1 - k))
        return np.array(ema_vals)

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    hist = macd - signal
    return {
        "macd":      round(float(macd[-1]), 4),
        "signal":    round(float(signal[-1]), 4),
        "histogram": round(float(hist[-1]), 4),
    }


def compute_bollinger_bands(prices: list, period: int = 20) -> dict:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return {"upper": None, "middle": None, "lower": None}
    closes = np.array([p["close"] for p in prices[-period:]])
    middle = np.mean(closes)
    std = np.std(closes)
    return {
        "upper":  round(float(middle + 2 * std), 2),
        "middle": round(float(middle), 2),
        "lower":  round(float(middle - 2 * std), 2),
    }


# ══════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
# ══════════════════════════════════════════════════════════════

def research_agent(symbol: str, question: str) -> str:
    """
    Research Agent — fetches news and market data,
    then uses Groq LLM to summarize findings.
    """
    quote = fetch_stock_quote(symbol)
    news = fetch_stock_news(symbol)

    context = f"""
Stock: {symbol}
Current Price: ${quote.get('price', 'N/A')}
Change: {quote.get('change_pct', 'N/A')}
Volume: {quote.get('volume', 'N/A')}

Recent News:
{json.dumps(news, indent=2)}
"""
    messages = [
        {
            "role": "system",
            "content": """You are a professional stock market research analyst.
Analyze the provided market data and news to answer the user's question.
Be concise, factual, and highlight the most important insights.
Always end with: '⚠️ This is not financial advice.'"""
        },
        {
            "role": "user",
            "content": f"Market data:\n{context}\n\nQuestion: {question}"
        }
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=500
    )
    return response.choices[0].message.content


def technical_agent(symbol: str, question: str) -> str:
    """
    Technical Analysis Agent — computes RSI, MACD,
    Bollinger Bands and interprets them.
    """
    prices = fetch_time_series(symbol, days=50)

    if not prices or "error" in prices[0]:
        return f"Could not fetch price data for {symbol}."

    rsi = compute_rsi(prices)
    macd = compute_macd(prices)
    bb = compute_bollinger_bands(prices)
    current_price = prices[-1]["close"] if prices else 0

    # Interpret RSI
    if rsi:
        if rsi > 70:
            rsi_signal = "OVERBOUGHT 🔴 — potential sell signal"
        elif rsi < 30:
            rsi_signal = "OVERSOLD 🟢 — potential buy signal"
        else:
            rsi_signal = "NEUTRAL ⚪ — no clear signal"
    else:
        rsi_signal = "Insufficient data"

    # Interpret MACD
    if macd["histogram"]:
        macd_signal = "BULLISH 🟢" if macd["histogram"] > 0 else "BEARISH 🔴"
    else:
        macd_signal = "Insufficient data"

    context = f"""
Symbol: {symbol}
Current Price: ${current_price}

Technical Indicators:
- RSI (14): {rsi} → {rsi_signal}
- MACD: {macd['macd']} | Signal: {macd['signal']} | Histogram: {macd['histogram']} → {macd_signal}
- Bollinger Bands: Upper ${bb['upper']} | Middle ${bb['middle']} | Lower ${bb['lower']}
  → Price vs BB: {"Above upper band ⚠️" if current_price > (bb['upper'] or 0)
                  else "Below lower band 🟢" if current_price < (bb['lower'] or 999999)
                  else "Within bands ✅"}
"""
    messages = [
        {
            "role": "system",
            "content": """You are an expert technical analyst.
Interpret the provided indicators and give a clear trading signal analysis.
Explain what each indicator means in simple terms.
Always end with: '⚠️ This is not financial advice.'"""
        },
        {
            "role": "user",
            "content": f"Technical data:\n{context}\n\nQuestion: {question}"
        }
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=500
    )
    return response.choices[0].message.content


def portfolio_agent(user_id: str, question: str) -> str:
    """
    Portfolio Agent — reads user's PostgreSQL portfolio,
    analyzes risk and suggests rebalancing.
    """
    portfolio = get_user_portfolio(user_id)
    prefs = get_user_preferences(user_id)

    if not portfolio:
        return "You don't have any stocks in your portfolio yet. Add stocks via the portfolio section."

    # Fetch current prices for all holdings
    holdings_data = []
    total_value = 0
    for holding in portfolio:
        quote = fetch_stock_quote(holding["symbol"])
        current_price = quote.get("price", holding["avg_cost"])
        current_value = current_price * holding["quantity"]
        cost_basis = holding["avg_cost"] * holding["quantity"]
        pnl = current_value - cost_basis
        pnl_pct = ((current_value - cost_basis) /
                   cost_basis * 100) if cost_basis else 0
        total_value += current_value
        holdings_data.append({
            "symbol":        holding["symbol"],
            "quantity":      holding["quantity"],
            "avg_cost":      holding["avg_cost"],
            "current_price": current_price,
            "current_value": round(current_value, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
        })

    # Calculate concentration %
    for h in holdings_data:
        h["concentration"] = round(
            (h["current_value"] / total_value * 100), 2) if total_value else 0

    context = f"""
User: {prefs.get('name', 'Investor')}
Risk Appetite: {prefs.get('riskAppetite', 'Unknown')}
Investment Goal: {prefs.get('investmentGoal', 'Unknown')}
Time Horizon: {prefs.get('timeHorizon', 'Unknown')}

Portfolio (Total Value: ${round(total_value, 2)}):
{json.dumps(holdings_data, indent=2)}
"""
    messages = [
        {
            "role": "system",
            "content": """You are a portfolio risk analyst.
Analyze the portfolio for concentration risk, P&L performance,
and alignment with the user's risk appetite and goals.
Suggest specific rebalancing actions if needed.
Always end with: '⚠️ This is not financial advice.'"""
        },
        {
            "role": "user",
            "content": f"Portfolio data:\n{context}\n\nQuestion: {question}"
        }
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=600
    )
    return response.choices[0].message.content


def sentiment_agent(symbol: str, question: str) -> str:
    """
    Sentiment Agent — scores news sentiment and
    gives overall bullish/bearish reading.
    """
    news = fetch_stock_news(symbol)

    if not news or "error" in news[0]:
        return f"Could not fetch news for {symbol}."

    scores = [float(a.get("score", 0)) for a in news if a.get("score")]
    avg_sentiment = round(sum(scores) / len(scores), 3) if scores else 0

    if avg_sentiment > 0.15:
        overall = "BULLISH 🟢"
    elif avg_sentiment < -0.15:
        overall = "BEARISH 🔴"
    else:
        overall = "NEUTRAL ⚪"

    context = f"""
Symbol: {symbol}
Average Sentiment Score: {avg_sentiment} (-1 bearish → +1 bullish)
Overall Reading: {overall}

News Articles:
{json.dumps(news, indent=2)}
"""
    messages = [
        {
            "role": "system",
            "content": """You are a market sentiment analyst specializing in NLP.
Analyze the news sentiment and explain what it means for the stock.
Identify key themes driving positive or negative sentiment.
Always end with: '⚠️ This is not financial advice.'"""
        },
        {
            "role": "user",
            "content": f"Sentiment data:\n{context}\n\nQuestion: {question}"
        }
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=400
    )
    return response.choices[0].message.content


def alert_agent(user_id: str, question: str) -> str:
    """
    Alert Agent — monitors portfolio stocks for
    RSI extremes, price vs moving average signals.
    """
    portfolio = get_user_portfolio(user_id)

    if not portfolio:
        return "No portfolio found to monitor for alerts."

    alerts = []
    for holding in portfolio:
        symbol = holding["symbol"]
        prices = fetch_time_series(symbol, days=30)
        if not prices or "error" in prices[0]:
            continue

        quote = fetch_stock_quote(symbol)
        rsi = compute_rsi(prices)
        bb = compute_bollinger_bands(prices)
        price = quote.get("price", 0)

        # Check alert conditions
        if rsi and rsi > 70:
            alerts.append(
                f"🔴 {symbol}: RSI={rsi} — OVERBOUGHT, consider taking profits")
        elif rsi and rsi < 30:
            alerts.append(
                f"🟢 {symbol}: RSI={rsi} — OVERSOLD, potential buying opportunity")

        if bb["upper"] and price > bb["upper"]:
            alerts.append(
                f"⚠️ {symbol}: Price ${price} broke above Bollinger upper band ${bb['upper']}")
        elif bb["lower"] and price < bb["lower"]:
            alerts.append(
                f"🟢 {symbol}: Price ${price} broke below Bollinger lower band ${bb['lower']}")

    if not alerts:
        alerts = [
            "✅ No critical alerts — all portfolio stocks are within normal ranges."]

    context = "\n".join(alerts)

    messages = [
        {
            "role": "system",
            "content": """You are a portfolio monitoring system.
Summarize the alerts clearly and suggest what action (if any) the investor should consider.
Be direct and actionable. Always end with: '⚠️ This is not financial advice.'"""
        },
        {
            "role": "user",
            "content": f"Current alerts:\n{context}\n\nUser question: {question}"
        }
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=400
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR — routes user message to right agent
# ══════════════════════════════════════════════════════════════

def detect_intent(message: str) -> dict:
    """
    Use Groq to detect which agent should handle this message.
    Returns: {"agent": "research|technical|portfolio|sentiment|alert|general", "symbol": "AAPL"}
    """
    prompt = f"""Analyze this stock market question and return a JSON object.

Question: "{message}"

Return ONLY valid JSON with these fields:
- "agent": one of "research", "technical", "portfolio", "sentiment", "alert", "general"
- "symbol": stock ticker symbol if mentioned (e.g. "AAPL"), or null

Rules:
- "research" → questions about company news, earnings, what's happening with a stock
- "technical" → RSI, MACD, charts, overbought, oversold, technical analysis
- "portfolio" → my portfolio, my holdings, rebalance, my stocks, P&L
- "sentiment" → sentiment, bullish, bearish, market mood, news feeling
- "alert" → alerts, warnings, monitor, threshold, should I be worried
- "general" → general market questions, greetings, other

Example: {{"agent": "technical", "symbol": "AAPL"}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    try:
        text = response.choices[0].message.content.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"agent": "general", "symbol": None}


def orchestrate(user_id: str, message: str) -> dict:
    """
    Main orchestrator — routes to specialist agent,
    saves conversation to PostgreSQL, returns response.
    """
    # Save user message
    save_message(user_id, "user", message)

    # Get past context for personalization
    history = get_chat_history(user_id, limit=6)
    memories = get_agent_memories(user_id, limit=3)
    prefs = get_user_preferences(user_id)

    # Detect intent
    intent = detect_intent(message)
    agent = intent.get("agent", "general")
    symbol = intent.get("symbol")

    # Route to specialist agent
    response_text = ""
    agent_used = agent

    if agent == "research" and symbol:
        response_text = research_agent(symbol, message)

    elif agent == "technical" and symbol:
        response_text = technical_agent(symbol, message)

    elif agent == "portfolio":
        response_text = portfolio_agent(user_id, message)

    elif agent == "sentiment" and symbol:
        response_text = sentiment_agent(symbol, message)

    elif agent == "alert":
        response_text = alert_agent(user_id, message)

    else:
        # General conversation with context
        history_text = "\n".join(
            [f"{h['role'].upper()}: {h['message']}" for h in history[-4:]]
        )
        memory_text = "\n".join([m["content"] for m in memories])

        messages = [
            {
                "role": "system",
                "content": f"""You are an AI stock market advisor.
User: {prefs.get('name', 'Investor')}
Risk appetite: {prefs.get('riskAppetite', 'unknown')}
Investment goal: {prefs.get('investmentGoal', 'unknown')}

Past context: {memory_text}

Be helpful, personalized, and concise.
Always end financial advice with: '⚠️ This is not financial advice.'"""
            }
        ]
        # Add conversation history
        for h in history[-4:]:
            messages.append({"role": h["role"], "content": h["message"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=500
        )
        response_text = response.choices[0].message.content
        agent_used = "general"

    # Save assistant response
    save_message(user_id, "assistant", response_text, agent_used)

    # Save important preferences mentioned
    keywords = ["prefer", "avoid", "risk", "goal", "invest", "horizon"]
    if any(k in message.lower() for k in keywords):
        save_agent_memory(user_id, message, memory_type="preference")

    return {
        "response":   response_text,
        "agent_used": agent_used,
        "symbol":     symbol,
        "intent":     intent,
    }
