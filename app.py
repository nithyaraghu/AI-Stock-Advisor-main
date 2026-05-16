from flask import Flask, request, jsonify, session
import requests
from flask_caching import Cache
from flask_cors import CORS
import logging
import json
import os
import numpy as np
import datetime
from datetime import timedelta
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings
from db import get_connection, get_cursor, init_db
from agents import orchestrate, get_chat_history, alert_agent

# Module-level constants
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
POLYGON_KEY  = os.environ.get('POLYGON_API_KEY', '')
FINNHUB_KEY  = os.environ.get('FINNHUB_API_KEY', '')

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
    # Only show warnings and errors - suppress yfinance debug spam
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('peewee').setLevel(logging.ERROR)


    @app.before_request
    def log_request_info():
        app.logger.debug('Request: %s %s', request.method, request.url)

    @app.after_request
    def log_response_info(response):
        app.logger.debug('Response: %s', response.status)
        return response

    init_db()

    @app.route('/')
    def home():
        return "<h1>AI Financial Advisor Backend</h1><p>The backend is running successfully.</p>"

    @app.errorhandler(404)
    def not_found(e):
        return "<h1>404 Not Found</h1>", 404

    @app.route('/debug/env')
    def debug_env():
        from agents import fetch_price_history
        import inspect
        src = inspect.getsource(fetch_price_history)
        has_polygon = 'POLYGON_KEY' in src
        return jsonify({
            'NEWS_API_KEY':    'SET' if os.environ.get('NEWS_API_KEY') else 'MISSING',
            'POLYGON_API_KEY': 'SET' if os.environ.get('POLYGON_API_KEY') else 'MISSING',
            'FINNHUB_API_KEY': 'SET' if os.environ.get('FINNHUB_API_KEY') else 'MISSING',
            'GROQ_API_KEY':    'SET' if os.environ.get('GROQ_API_KEY') else 'MISSING',
            'DATABASE_URL':    'SET' if os.environ.get('DATABASE_URL') else 'MISSING',
            'agents_has_polygon_history': has_polygon,
        })

    @app.route('/debug/history')
    def debug_history():
        """Test what fetch_price_history returns for AAPL."""
        try:
            from agents import fetch_price_history
            data = fetch_price_history('AAPL', '3mo')
            return jsonify({
                'count': len(data),
                'first': data[0] if data else None,
                'last': data[-1] if data else None,
                'polygon_key': 'SET' if os.environ.get('POLYGON_API_KEY') else 'MISSING',
            })
        except Exception as e:
            return jsonify({'error': str(e)})

    @app.route('/debug/polygon')
    def debug_polygon():
        """Test Polygon API response."""
        if not POLYGON_KEY:
            return jsonify({'error': 'POLYGON_API_KEY not set'})
        try:
            # Test prev aggs
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={POLYGON_KEY}"
            r = requests.get(url, timeout=10).json()
            return jsonify({'prev_aggs': r})
        except Exception as e:
            return jsonify({'error': str(e)})

    @app.route('/signup', methods=['POST'])
    def signup():
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                return jsonify({"message": "Email already exists"}), 400
            cur.execute("""
                INSERT INTO users (email, password, name, preferences)
                VALUES (%s, %s, %s, %s)
            """, (email, hashed_password, username, json.dumps({
                "gender": data.get('gender'), "age": data.get('age'),
                "investmentGoal": data.get('investmentGoal'),
                "riskAppetite": data.get('riskAppetite'),
                "timeHorizon": data.get('timeHorizon')
            })))
            conn.commit()
            return jsonify({"message": "User signed up successfully"}), 201
        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error creating user", "error": str(e)}), 500
        finally:
            cur.close(); conn.close()

    @app.route('/login', methods=['POST'])
    def login():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 400
            try:
                pwd_match = bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8'))
            except Exception:
                pwd_match = False
            if not pwd_match:
                return jsonify({"message": "Invalid email or password"}), 400
            return jsonify({"message": "Login successful", "email": user['email'], "username": user['name']}), 200
        except Exception as e:
            return jsonify({"message": "Error during login", "error": str(e)}), 500
        finally:
            cur.close(); conn.close()

    @app.route('/user-details/<email>', methods=['GET'])
    def get_user_details(email):
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 404
            prefs = user.get('preferences', {})
            cur.execute("SELECT symbol, quantity, avg_cost FROM portfolio_holdings WHERE user_id = %s", (str(user['id']),))
            portfolio = cur.fetchall()
            return jsonify({
                "userId": str(user['id']), "email": user['email'], "username": user['name'],
                "gender": prefs.get('gender'), "age": prefs.get('age'),
                "investmentGoal": prefs.get('investmentGoal'), "riskAppetite": prefs.get('riskAppetite'),
                "timeHorizon": prefs.get('timeHorizon'), "portfolio": [dict(p) for p in portfolio]
            }), 200
        except Exception as e:
            return jsonify({"message": "Error retrieving user details", "error": str(e)}), 500
        finally:
            cur.close(); conn.close()

    @app.route('/user/<email>/portfolio', methods=['POST'])
    def add_stock_to_portfolio(email):
        data = request.get_json()
        new_stock = data.get('stock')
        if not new_stock:
            return jsonify({"message": "No stock data provided"}), 400
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 404
            cur.execute("""
                INSERT INTO portfolio_holdings (user_id, symbol, quantity, avg_cost)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, symbol)
                DO UPDATE SET quantity = EXCLUDED.quantity, avg_cost = EXCLUDED.avg_cost, updated_at = NOW()
            """, (str(user['id']), new_stock.get('symbol'), new_stock.get('quantity', 0), new_stock.get('avg_cost', 0)))
            conn.commit()
            return jsonify({"message": "Stock added to portfolio successfully"}), 200
        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error updating portfolio", "error": str(e)}), 500
        finally:
            cur.close(); conn.close()

    @app.route('/portfolio/<email>/<symbol>', methods=['DELETE'])
    def delete_holding(email, symbol):
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 404
            conn.cursor().execute("DELETE FROM portfolio_holdings WHERE user_id = %s AND symbol = %s", (str(user['id']), symbol.upper()))
            conn.commit()
            return jsonify({"message": f"{symbol} removed from portfolio"}), 200
        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error removing stock", "error": str(e)}), 500
        finally:
            cur.close(); conn.close()

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,DELETE')
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cache-Control'] = 'no-store'
        return response

    # -------------- yfinance Stock Routes --------------

    @app.route('/stocks/yf/quote', methods=['GET'])
    def get_yf_quote():
        """Get live stock quote — Finnhub first (60 req/min), Polygon fallback, yfinance last."""
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide symbol'}), 400
        try:
            sym = symbol.upper()

            # 1. Finnhub — 60 req/min free, no IP blocking
            if FINNHUB_KEY:
                try:
                    r = requests.get(
                        f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}",
                        timeout=10).json()
                    price = float(r.get('c', 0) or 0)
                    prev  = float(r.get('pc', 0) or 0)
                    if price > 0:
                        return jsonify({
                            'symbol':     sym,
                            'price':      round(price, 2),
                            'prev_close': round(prev, 2),
                            'change_pct': round(((price-prev)/prev*100), 2) if prev else 0,
                            'high':       round(float(r.get('h', 0) or 0), 2),
                            'low':        round(float(r.get('l', 0) or 0), 2),
                            'volume':     0,
                        })
                except Exception:
                    pass

            # 2. Polygon fallback
            if POLYGON_KEY:
                try:
                    from datetime import date, timedelta
                    end   = date.today()
                    start = end - timedelta(days=10)
                    url   = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
                             f"{start}/{end}?adjusted=true&sort=desc&limit=2&apiKey={POLYGON_KEY}")
                    r = requests.get(url, timeout=15).json()
                    results = r.get('results', [])
                    if results:
                        latest = results[0]
                        prev_r = results[1] if len(results) > 1 else results[0]
                        price  = float(latest.get('c', 0) or 0)
                        prev   = float(prev_r.get('c', price) or price)
                        if price > 0:
                            return jsonify({
                                'symbol':     sym,
                                'price':      round(price, 2),
                                'prev_close': round(prev, 2),
                                'change_pct': round(((price-prev)/prev*100), 2) if prev else 0,
                                'high':       round(float(latest.get('h', 0)), 2),
                                'low':        round(float(latest.get('l', 0)), 2),
                                'volume':     int(latest.get('v', 0)),
                            })
                except Exception:
                    pass

            # 3. yfinance last resort
            try:
                import yfinance as yf
                info  = yf.Ticker(sym).fast_info
                price = float(info.last_price or 0)
                prev  = float(info.previous_close or 0)
                if price > 0:
                    return jsonify({
                        'symbol':     sym,
                        'price':      round(price, 2),
                        'prev_close': round(prev, 2),
                        'change_pct': round(((price-prev)/prev*100), 2) if prev else 0,
                        'high':       round(float(info.day_high or 0), 2),
                        'low':        round(float(info.day_low or 0), 2),
                        'volume':     int(info.three_month_average_volume or 0),
                    })
            except Exception:
                pass

            return jsonify({'symbol': sym, 'price': 0, 'prev_close': 0, 'change_pct': 0})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/stocks/yf/history', methods=['GET'])
    def get_yf_history():
        symbol   = request.args.get('symbol')
        period   = request.args.get('period', '3mo')
        interval = request.args.get('interval', '1d')
        if not symbol:
            return jsonify({'error': 'Please provide symbol'}), 400
        try:
            if POLYGON_KEY:
                # Convert period to date range
                from datetime import date, timedelta
                period_days = {
                    '5d': 5, '1mo': 30, '3mo': 90,
                    '6mo': 180, '1y': 365, '2y': 730
                }
                days = period_days.get(period, 90)
                end   = date.today()
                start = end - timedelta(days=days)
                # Map interval
                timespan = 'day' if interval == '1d' else 'week' if interval == '1wk' else 'hour'
                url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/{timespan}/"
                       f"{start}/{end}?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_KEY}")
                r = requests.get(url, timeout=15).json()
                results = r.get('results', [])
                if not results:
                    return jsonify({'error': 'No data found'}), 404
                data = [{
                    'time':   datetime.datetime.fromtimestamp(d['t']/1000).strftime('%Y-%m-%d'),
                    'open':   round(d.get('o', 0), 2),
                    'high':   round(d.get('h', 0), 2),
                    'low':    round(d.get('l', 0), 2),
                    'close':  round(d.get('c', 0), 2),
                    'volume': int(d.get('v', 0)),
                } for d in results]
                return jsonify({'symbol': symbol.upper(), 'data': data})
            # Fallback to yfinance
            import yfinance as yf
            hist = yf.Ticker(symbol.upper()).history(period=period, interval=interval)
            if hist.empty:
                return jsonify({'error': 'No data found'}), 404
            data = [{'time': str(date)[:10], 'open': round(float(row['Open']), 2),
                     'high': round(float(row['High']), 2), 'low': round(float(row['Low']), 2),
                     'close': round(float(row['Close']), 2), 'volume': int(row['Volume'])}
                    for date, row in hist.iterrows()]
            return jsonify({'symbol': symbol.upper(), 'data': data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/stocks/yf/multi', methods=['GET'])
    def get_yf_multi():
        """Get quotes for multiple symbols using Finnhub."""
        symbols = request.args.get('symbols', '')
        if not symbols:
            return jsonify({'error': 'Please provide symbols'}), 400
        try:
            sym_list = symbols.upper().split()
            result   = {}
            if FINNHUB_KEY:
                for sym in sym_list:
                    try:
                        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}"
                        r   = requests.get(url, timeout=10).json()
                        price = r.get('c', 0)
                        prev  = r.get('pc', 0)
                        chg   = round(((price - prev) / prev * 100), 2) if prev else 0
                        result[sym] = {
                            'price':      round(float(price), 2),
                            'prev_close': round(float(prev), 2),
                            'change_pct': chg,
                        }
                    except:
                        result[sym] = {'price': 0, 'prev_close': 0, 'change_pct': 0}
                return jsonify(result)
            # Fallback to Polygon
            if POLYGON_KEY:
                for sym in sym_list:
                    try:
                        url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?apiKey={POLYGON_KEY}"
                        r   = requests.get(url, timeout=10).json()
                        res = r.get('results', [{}])[0] if r.get('results') else {}
                        price = res.get('c', 0); prev = res.get('o', price)
                        result[sym] = {'price': round(float(price), 2), 'prev_close': round(float(prev), 2),
                                       'change_pct': round(((price-prev)/prev*100), 2) if prev else 0}
                    except:
                        result[sym] = {'price': 0, 'prev_close': 0, 'change_pct': 0}
                return jsonify(result)
            # Fallback to yfinance
            import yfinance as yf
            tickers = yf.Tickers(symbols.upper())
            for sym in sym_list:
                try:
                    info = tickers.tickers[sym].fast_info
                    result[sym] = {
                        'price':      round(float(info.last_price or 0), 2),
                        'prev_close': round(float(info.previous_close or 0), 2),
                        'change_pct': round(((info.last_price-info.previous_close)/info.previous_close*100), 2) if info.previous_close else 0,
                    }
                except:
                    result[sym] = {'price': 0, 'prev_close': 0, 'change_pct': 0}
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # -------------- News Routes (NewsAPI) --------------

    COMPANY_NAMES = {
        'AAPL':'Apple','NVDA':'NVIDIA','MSFT':'Microsoft','GOOGL':'Alphabet Google',
        'META':'Meta Platforms','AMZN':'Amazon','TSLA':'Tesla','AMD':'AMD',
        'INTC':'Intel','JPM':'JPMorgan','BAC':'Bank of America','V':'Visa',
        'MA':'Mastercard','NFLX':'Netflix','DIS':'Disney','COIN':'Coinbase',
        'PYPL':'PayPal','UBER':'Uber','SHOP':'Shopify','PLTR':'Palantir',
        'SPY':'S&P 500','QQQ':'Nasdaq','GLD':'Gold',
    }

    @app.route('/stocks/news/market', methods=['GET'])
    def get_market_news():
        try:
            if NEWS_API_KEY:
                url = f"https://newsapi.org/v2/everything?q=stock+market+wall+street&sortBy=publishedAt&pageSize=15&apiKey={NEWS_API_KEY}&language=en"
                data = requests.get(url, timeout=10).json()
                articles = [{'title': a.get('title',''), 'url': a.get('url',''),
                             'source': a.get('source',{}).get('name',''),
                             'thumbnail': a.get('urlToImage',''),
                             'published': a.get('publishedAt',''), 'symbol': 'MARKET'}
                            for a in data.get('articles',[])
                            if a.get('title') and '[Removed]' not in a.get('title','')]
                return jsonify({'articles': articles})
            return jsonify({'articles': []})
        except Exception as e:
            return jsonify({'articles': [], 'error': str(e)}), 200

    @app.route('/stocks/news/ticker', methods=['GET'])
    def get_ticker_headlines():
        try:
            symbols = ['AAPL','NVDA','MSFT','GOOGL','TSLA','META','AMZN','SPY','JPM','NFLX']
            headlines = []
            if NEWS_API_KEY:
                url = f"https://newsapi.org/v2/everything?q=stocks+market&sortBy=publishedAt&pageSize=20&apiKey={NEWS_API_KEY}&language=en"
                data = requests.get(url, timeout=10).json()
                for i, a in enumerate(data.get('articles', [])[:20]):
                    if a.get('title') and '[Removed]' not in a.get('title',''):
                        headlines.append({'symbol': symbols[i % len(symbols)],
                                         'title': a.get('title','')[:100], 'url': a.get('url','#')})
            return jsonify({'headlines': headlines})
        except Exception as e:
            return jsonify({'headlines': []}), 200

    @app.route('/stocks/yf/news', methods=['GET'])
    def get_yf_news():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide symbol'}), 400
        try:
            if NEWS_API_KEY:
                query = COMPANY_NAMES.get(symbol.upper(), symbol)
                url = f"https://newsapi.org/v2/everything?q={query}+stock&sortBy=publishedAt&pageSize=8&apiKey={NEWS_API_KEY}&language=en"
                data = requests.get(url, timeout=10).json()
                articles = [{'title': a.get('title',''), 'url': a.get('url',''),
                             'source': a.get('source',{}).get('name',''),
                             'thumbnail': a.get('urlToImage',''),
                             'published': a.get('publishedAt',''),
                             'symbol': symbol.upper(),
                             'overall_sentiment_label': 'Neutral',
                             'overall_sentiment_score': '0'}
                            for a in data.get('articles',[])
                            if a.get('title') and '[Removed]' not in a.get('title','')]
                return jsonify({'articles': articles})
            return jsonify({'articles': []})
        except Exception as e:
            return jsonify({'articles': [], 'error': str(e)}), 200

    # -------------- Chat Routes --------------

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        user_id = data.get('user_id')
        message = data.get('message')
        if not message:
            return jsonify({"error": "message is required"}), 400
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        if not user_id or not uuid_pattern.match(str(user_id)):
            user_id = "00000000-0000-0000-0000-000000000001"
        try:
            import traceback
            result = orchestrate(user_id, message)
            return jsonify({"response": result["response"], "agent_used": result["agent_used"], "symbol": result["symbol"]}), 200
        except Exception as e:
            tb = traceback.format_exc()
            app.logger.error(f"Chat error: {e}\n{tb}")
            return jsonify({"error": str(e)}), 500

    @app.route('/chat/history/<user_id>', methods=['GET'])
    def chat_history(user_id):
        try:
            history = get_chat_history(user_id, limit=20)
            return jsonify({"history": history}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/chat/alerts/<user_id>', methods=['GET'])
    def get_alerts(user_id):
        try:
            alerts = alert_agent(user_id, "Check my portfolio for any alerts")
            return jsonify({"alerts": alerts}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)