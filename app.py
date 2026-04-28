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

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
    logging.basicConfig(level=logging.DEBUG)

    # ------------- Logging ------------------
    @app.before_request
    def log_request_info():
        app.logger.debug('--- Incoming Request ---')
        app.logger.debug('Request Method: %s', request.method)
        app.logger.debug('Request URL: %s', request.url)

    @app.after_request
    def log_response_info(response):
        app.logger.debug('--- Outgoing Response ---')
        app.logger.debug('Response Status: %s', response.status)
        return response

    # -------------- PostgreSQL Init --------------
    # Replaces Cosmos DB initialization
    init_db()

    # API Key
    API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '9ZQUXAH9JOQRSQDV')

    # -------------- Auth Routes --------------

    @app.route('/')
    def home():
        return "<h1>AI Financial Advisor Backend</h1><p>The backend is running successfully.</p>"

    @app.errorhandler(404)
    def not_found(e):
        return "<h1>404 Not Found</h1><p>The requested resource could not be found.</p>", 404

    @app.route('/signup', methods=['POST'])
    def signup():
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        gender = data.get('gender')
        age = data.get('age')
        investment_goal = data.get('investmentGoal')
        risk_appetite = data.get('riskAppetite')
        time_horizon = data.get('timeHorizon')

        # Hash the password
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt()
        ).decode('utf-8')

        conn = get_connection()
        cur = get_cursor(conn)
        try:
            # Check if user already exists
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                return jsonify({"message": "Email already exists"}), 400

            # Insert new user
            cur.execute("""
                INSERT INTO users
                    (email, password, name, preferences)
                VALUES (%s, %s, %s, %s)
            """, (
                email,
                hashed_password,
                username,
                json.dumps({
                    "gender": gender,
                    "age": age,
                    "investmentGoal": investment_goal,
                    "riskAppetite": risk_appetite,
                    "timeHorizon": time_horizon
                })
            ))
            conn.commit()
            return jsonify({"message": "User signed up successfully"}), 201

        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error creating user", "error": str(e)}), 500
        finally:
            cur.close()
            conn.close()

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

            if not bcrypt.checkpw(
                password.encode('utf-8'),
                user['password'].encode('utf-8')
            ):
                return jsonify({"message": "Invalid credentials"}), 400

            return jsonify({
                "message": "Login successful",
                "email": user['email'],
                "username": user['name'],
            }), 200

        except Exception as e:
            return jsonify({"message": "Error during login", "error": str(e)}), 500
        finally:
            cur.close()
            conn.close()

    # -------------- User Account Routes --------------

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

            # Get portfolio
            cur.execute(
                "SELECT symbol, quantity, avg_cost FROM portfolio_holdings WHERE user_id = %s",
                (str(user['id']),)
            )
            portfolio = cur.fetchall()

            return jsonify({
                "email": user['email'],
                "username": user['name'],
                "gender": prefs.get('gender'),
                "age": prefs.get('age'),
                "investmentGoal": prefs.get('investmentGoal'),
                "riskAppetite": prefs.get('riskAppetite'),
                "timeHorizon": prefs.get('timeHorizon'),
                "portfolio": [dict(p) for p in portfolio]
            }), 200

        except Exception as e:
            return jsonify({"message": "Error retrieving user details", "error": str(e)}), 500
        finally:
            cur.close()
            conn.close()

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

            # Upsert portfolio holding
            cur.execute("""
                INSERT INTO portfolio_holdings (user_id, symbol, quantity, avg_cost)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, symbol)
                DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    avg_cost = EXCLUDED.avg_cost,
                    updated_at = NOW()
            """, (
                str(user['id']),
                new_stock.get('symbol'),
                new_stock.get('quantity', 0),
                new_stock.get('avg_cost', 0)
            ))
            conn.commit()
            return jsonify({"message": "Stock added to portfolio successfully"}), 200

        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error updating portfolio", "error": str(e)}), 500
        finally:
            cur.close()
            conn.close()

    @app.route('/delete-portfolio/<email>', methods=['DELETE'])
    def delete_portfolio(email):
        conn = get_connection()
        cur = get_cursor(conn)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 404

            cur.execute(
                "DELETE FROM portfolio_holdings WHERE user_id = %s",
                (str(user['id']),)
            )
            conn.commit()
            return jsonify({"message": "User portfolio deleted successfully"}), 200

        except Exception as e:
            conn.rollback()
            return jsonify({"message": "Error deleting portfolio", "error": str(e)}), 500
        finally:
            cur.close()
            conn.close()

    # -------------- Stock Routes --------------

    @cache.cached()
    @app.route('/stocks/quote', methods=['GET'])
    def get_stock_quote():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    @cache.cached()
    @app.route('/stocks/overview', methods=['GET'])
    def get_stock_overview():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'OVERVIEW', 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    @cache.cached()
    @app.route('/stocks/income_statement', methods=['GET'])
    def get_income_statement():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'INCOME_STATEMENT', 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    @cache.cached()
    @app.route('/stocks/news', methods=['GET'])
    def get_stock_news():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'NEWS_SENTIMENT', 'tickers': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    @cache.cached()
    @app.route('/stocks/insider_transactions', methods=['GET'])
    def get_insider_transactions():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'INSIDER_TRANSACTIONS', 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    @app.route('/stocks/time_series_monthly', methods=['GET'])
    @cache.cached(timeout=300, query_string=True)
    def get_stock_time_series_monthly():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': 'TIME_SERIES_MONTHLY_ADJUSTED', 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        time_series_key = 'Monthly Adjusted Time Series'
        if time_series_key not in data:
            return jsonify({'error': 'No data found for the requested time series.'}), 400

        processed_data = [
            {
                'time': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'adjusted_close': float(values['5. adjusted close']),
                'volume': int(values['6. volume']),
                'dividend_amount': float(values['7. dividend amount']),
            }
            for date, values in data[time_series_key].items()
        ]
        processed_data.sort(key=lambda x: x['time'])
        return jsonify({'symbol': symbol, 'data': processed_data})

    @app.route('/stocks/time_series', methods=['GET'])
    @cache.cached()
    def get_stock_time_series():
        symbol = request.args.get('symbol')
        time_series_function = request.args.get('function', 'TIME_SERIES_DAILY')

        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400
        if time_series_function not in ['TIME_SERIES_DAILY', 'TIME_SERIES_WEEKLY', 'TIME_SERIES_MONTHLY']:
            return jsonify({'error': f'Invalid time series function: {time_series_function}'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {'function': time_series_function, 'symbol': symbol.upper(), 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        time_series_key = {
            'TIME_SERIES_DAILY': 'Time Series (Daily)',
            'TIME_SERIES_WEEKLY': 'Weekly Time Series',
            'TIME_SERIES_MONTHLY': 'Monthly Time Series'
        }.get(time_series_function)

        if time_series_key not in data:
            return jsonify({'error': 'No data found for the requested time series.'}), 400

        processed_data = [
            {
                'time': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume']),
            }
            for date, values in data[time_series_key].items()
        ]
        processed_data.sort(key=lambda x: x['time'])
        return jsonify({'symbol': symbol, 'data': processed_data})

    @cache.cached()
    @app.route('/stocks/daily', methods=['GET'])
    def get_stock_daily():
        symbol = request.args.get('symbol')
        outputsize = request.args.get('outputsize', 'compact')
        datatype = request.args.get('datatype', 'json')

        if not symbol:
            return jsonify({'error': 'Please provide the stock symbol.'}), 400

        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol.upper(),
            'outputsize': outputsize,
            'datatype': datatype,
            'apikey': API_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'Error Message' in data or 'Note' in data:
            return jsonify({'error': data.get('Error Message') or data.get('Note')}), 400
        return jsonify(data)

    # -------------- Top Stocks Routes --------------

    @cache.cached()
    @app.route('/stocks/top_movers', methods=['GET'])
    def get_top_movers():
        url = 'https://www.alphavantage.co/query'
        params = {'function': 'TOP_GAINERS_LOSERS', 'apikey': API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

        data = response.json()
        if 'top_gainers' in data and 'top_losers' in data and 'most_actively_traded' in data:
            combined_data = {
                'top_gainers': data['top_gainers'][:10],
                'top_losers': data['top_losers'][:10],
                'most_active': data['most_actively_traded'][:10]
            }

            # Cache top movers in PostgreSQL price_cache
            conn = get_connection()
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO price_cache (symbol, ohlcv, fetched_at)
                    VALUES ('TOP_MOVERS', %s, NOW())
                    ON CONFLICT (symbol) DO UPDATE
                    SET ohlcv = EXCLUDED.ohlcv, fetched_at = NOW()
                """, (json.dumps(combined_data),))
                conn.commit()
            except Exception as e:
                app.logger.error('Error caching top movers: %s', str(e))
            finally:
                cur.close()
                conn.close()

            return jsonify(combined_data)
        else:
            return jsonify({'error': 'No data found for top movers.'}), 400

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,DELETE')
        return response

    # -------------- ML Processing Routes --------------

    @app.route('/api/rank-news', methods=['POST'])
    def rank_news():
        data = request.get_json()
        news_articles = data.get('newsArticles', [])
        if not news_articles:
            return jsonify({'error': 'No news articles provided.'}), 400

        ranked_articles = rank_news_by_impact(news_articles)
        return jsonify(ranked_articles), 200

    def rank_news_by_impact(news_articles):
        SENTIMENT_WEIGHT = 0.5
        RECENCY_WEIGHT = 0.3
        SOURCE_WEIGHT = 0.2
        current_time = datetime.datetime.utcnow()
        source_credibility = {
            'Reuters': 1.0, 'Bloomberg': 0.9,
            'Wall Street Journal': 0.9, 'CNBC': 0.8,
            'Yahoo Finance': 0.7, 'Motley Fool': 0.6,
            'Seeking Alpha': 0.6, 'Benzinga': 0.5,
        }
        ranked_articles = []
        for idx, article in enumerate(news_articles):
            try:
                sentiment_score = abs(float(article.get('overall_sentiment_score', 0)))
                time_published = article.get('time_published')
                try:
                    article_time = datetime.datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                    time_diff = (current_time - article_time).total_seconds() / 3600
                    recency_score = 1 / (1 + time_diff)
                except ValueError:
                    recency_score = 0

                source = article.get('source', '').strip()
                credibility_score = source_credibility.get(source, 0.5)
                impact_score = (SENTIMENT_WEIGHT * sentiment_score +
                                RECENCY_WEIGHT * recency_score +
                                SOURCE_WEIGHT * credibility_score)
                ranked_article = article.copy()
                ranked_article['impact_score'] = impact_score
                ranked_articles.append(ranked_article)
            except Exception as e:
                app.logger.error(f"Error processing article {idx}: {e}")
                continue

        return sorted(ranked_articles, key=lambda x: x.get('impact_score', 0), reverse=True)


    # -------------- yfinance Routes (unlimited, no API key) --------------

    @app.route('/stocks/yf/quote', methods=['GET'])
    def get_yf_quote():
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please provide symbol'}), 400
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.upper())
            info = ticker.fast_info
            return jsonify({
                'symbol':     symbol.upper(),
                'price':      round(float(info.last_price or 0), 2),
                'prev_close': round(float(info.previous_close or 0), 2),
                'change_pct': round(((info.last_price - info.previous_close) / info.previous_close * 100), 2) if info.previous_close else 0,
                'volume':     int(info.three_month_average_volume or 0),
                'high':       round(float(info.day_high or 0), 2),
                'low':        round(float(info.day_low or 0), 2),
            })
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
            import yfinance as yf
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                return jsonify({'error': 'No data found'}), 404
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'time':   str(date)[:10],
                    'open':   round(float(row['Open']), 2),
                    'high':   round(float(row['High']), 2),
                    'low':    round(float(row['Low']), 2),
                    'close':  round(float(row['Close']), 2),
                    'volume': int(row['Volume']),
                })
            return jsonify({'symbol': symbol.upper(), 'data': data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/stocks/yf/multi', methods=['GET'])
    def get_yf_multi():
        symbols = request.args.get('symbols', '')
        if not symbols:
            return jsonify({'error': 'Please provide symbols'}), 400
        try:
            import yfinance as yf
            tickers = yf.Tickers(symbols.upper())
            result = {}
            for sym in symbols.upper().split():
                try:
                    info = tickers.tickers[sym].fast_info
                    result[sym] = {
                        'price':      round(float(info.last_price or 0), 2),
                        'prev_close': round(float(info.previous_close or 0), 2),
                        'change_pct': round(((info.last_price - info.previous_close) / info.previous_close * 100), 2) if info.previous_close else 0,
                    }
                except:
                    result[sym] = {'price': 0, 'prev_close': 0, 'change_pct': 0}
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # -------------- Agentic Chat Routes --------------

    @app.route('/chat', methods=['POST'])
    def chat():
        """
        Main chat endpoint — routes to specialist agents.
        Body: {"user_id": "uuid", "message": "your question"}
        """
        data    = request.get_json()
        user_id = data.get('user_id')
        message = data.get('message')

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        try:
            result = orchestrate(user_id, message)
            return jsonify({
                "response":   result["response"],
                "agent_used": result["agent_used"],
                "symbol":     result["symbol"],
            }), 200
        except Exception as e:
            app.logger.error(f"Chat error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/chat/history/<user_id>', methods=['GET'])
    def chat_history(user_id):
        """Get conversation history for a user."""
        try:
            history = get_chat_history(user_id, limit=20)
            return jsonify({"history": history}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/chat/alerts/<user_id>', methods=['GET'])
    def get_alerts(user_id):
        """Get proactive portfolio alerts for a user."""
        try:
            alerts = alert_agent(user_id, "Check my portfolio for any alerts")
            return jsonify({"alerts": alerts}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# Create the Flask app
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)