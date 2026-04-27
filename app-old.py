from flask import Flask, request, jsonify, session
import requests
from flask_caching import Cache
from flask_cors import CORS
import logging
from sentence_transformers import SentenceTransformer
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
import chromadb
from chromadb.config import Settings
import json
import os
import numpy as np
import datetime 
from datetime import timedelta
import bcrypt

# Azure Cosmos DB
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import datetime

from config import settings
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)

# Configure JWT
app.config['JWT_SECRET_KEY'] = 'AI-Financial-Advisor-151124'  # Replace with a secure secret key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300}) 
logging.basicConfig(level=logging.DEBUG)
# Initializing Embeddings model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the ChromaDB client with updated settings
chroma_settings = Settings(persist_directory="chroma_data")

client = chromadb.Client(chroma_settings)

collection = client.get_or_create_collection("stock_data")

# ------------ Hashing Passwords ------------
# from werkzeug.security import generate_password_hash, check_password_hash

# # Storing a hashed password
# hashed_password = generate_password_hash(password)

# # Verifying a password
# check_password_hash(hashed_password, password)

# ------------- Logging ------------------ 

@app.before_request
def log_request_info():
    app.logger.debug('--- Incoming Request ---')
    app.logger.debug('Request Method: %s', request.method)
    app.logger.debug('Request URL: %s', request.url)
    app.logger.debug('Request Headers: %s', request.headers)
    app.logger.debug('Request Body: %s', request.get_data())

@app.after_request
def log_response_info(response):
    app.logger.debug('--- Outgoing Response ---')
    app.logger.debug('Response Status: %s', response.status)
    app.logger.debug('Response Headers: %s', response.headers)
    app.logger.debug('Response Body: %s', response.get_data(as_text=True))
    return response

#  -------------- Azure Initialization --------------
# Azure Cosmos DB settings
HOST = settings['host']
MASTER_KEY = settings['master_key']
DATABASE_ID = settings['database_id']
CONTAINER_ID = settings['container_id']

# Initialize the Cosmos client
cosmos_client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY})

# Get or create the database
try:
    db = cosmos_client.create_database(id=DATABASE_ID)
    print('Database with id \'{0}\' created'.format(DATABASE_ID))
except exceptions.CosmosResourceExistsError:
    db = cosmos_client.get_database_client(DATABASE_ID)
    print('Database with id \'{0}\' was found'.format(DATABASE_ID))

# Get or create the container
try:
    container = db.create_container(id=CONTAINER_ID, partition_key=PartitionKey(path='/partitionKey'))
    print('Container with id \'{0}\' created'.format(CONTAINER_ID))
except exceptions.CosmosResourceExistsError:
    container = db.get_container_client(CONTAINER_ID)
    print('Container with id \'{0}\' was found'.format(CONTAINER_ID))


# -------------- Auth Routes --------------

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    gender = data.get('gender')
    age = data.get('age')
    investment_amount = data.get('investmentAmount')
    investment_goal = data.get('investmentGoal')
    risk_appetite = data.get('riskAppetite')
    time_horizon = data.get('timeHorizon')

    # Check if user already exists
    try:
        existing_user = list(container.query_items(
            query=f"SELECT * FROM c WHERE c.email = '{email}'", enable_cross_partition_query=True
        ))
        if existing_user:
            return jsonify({"message": "Email already exists"}), 400
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error checking user existence", "error": str(e)}), 500

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Create new user document
    new_user = {
        "id": email,  # Using email as the unique identifier
        "username": username,
        "email": email,
        "password": hashed_password.decode('utf-8'),
        "gender": gender,
        "age": age,
        "investmentAmount": investment_amount,
        "investmentGoal": investment_goal,
        "riskAppetite": risk_appetite,
        "timeHorizon": time_horizon
    }

    try:
        # Insert new user into Cosmos DB container
        container.create_item(body=new_user)
        # Generate JWT token
        access_token = create_access_token(identity=email)
        response = jsonify({"message": "User signed up successfully"})
        response.set_cookie('access_token', access_token, httponly=True)
        return jsonify({"message": "User signed up successfully"}), 201
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error creating user", "error": str(e)}), 500

# Login Route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Retrieve user by email
    try:
        user = list(container.query_items(
            query=f"SELECT * FROM c WHERE c.email = '{email}'", enable_cross_partition_query=True
        ))
        if not user:
            return jsonify({"message": "User not found"}), 400
        user = user[0]  # There should be only one user with a unique email
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error retrieving user", "error": str(e)}), 500

    # Check if password matches
    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({"message": "Invalid credentials"}), 400

    # Return user details in the response
    response_data = {
        "message": "Login successful",
        "email": user['email'],
        "username": user['username'],
        # Include other user details if needed
    }
    return jsonify(response_data), 200


def generate_embedding(text):
    return embedding_model.encode([text])[0].tolist()


API_KEY = '9ZQUXAH9JOQRSQDV'

# -------------- User Account Routes --------------

@app.route('/user-details/<email>', methods=['GET'])
def get_user_details(email):
    # Retrieve user by email
    try:
        user = list(container.query_items(
            query=f"SELECT * FROM c WHERE c.email = '{email}'", enable_cross_partition_query=True
        ))
        if not user:
            return jsonify({"message": "User not found"}), 404
        user = user[0]  # There should be only one user with a unique email
        # Exclude sensitive data like password
        user_details = {
            "email": user['email'],
            "username": user['username'],
            "gender": user['gender'],
            "age": user['age'],
            "investmentGoal": user['investmentGoal'],
            "riskAppetite": user['riskAppetite'],
            "timeHorizon": user['timeHorizon'],
            "portfolio": user.get('portfolio', [])  # Include portfolio
        }
        return jsonify(user_details), 200
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error retrieving user details", "error": str(e)}), 500

@app.route('/user/<email>/portfolio', methods=['POST'])
def add_stock_to_portfolio(email):
    data = request.get_json()
    new_stock = data.get('stock')

    if not new_stock:
        return jsonify({"message": "No stock data provided"}), 400

    try:
        # Retrieve user by email
        user = list(container.query_items(
            query=f"SELECT * FROM c WHERE c.email = '{email}'",
            enable_cross_partition_query=True
        ))
        if not user:
            return jsonify({"message": "User not found"}), 404

        user = user[0]  # Assuming email is unique

        # Update user's portfolio
        portfolio = user.get('portfolio', [])
        portfolio.append(new_stock)
        user['portfolio'] = portfolio

        # Replace the user document in the database
        container.replace_item(item=user, body=user)

        return jsonify({"message": "Stock added to portfolio successfully", "portfolio": portfolio}), 200

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error updating portfolio", "error": str(e)}), 500

# -------------- Stock Routes --------------

@cache.cached()
@app.route('/stocks/quote', methods=['GET'])
def get_stock_quote():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol.upper(),
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)  # Convert data to string
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol}],
        ids=[symbol]  # Use the stock symbol as the ID
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/overview', methods=['GET'])
def get_stock_overview():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol.upper(),
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)  # Convert data to string
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol}],
        ids=[f"{symbol}_overview"]
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/income_statement', methods=['GET'])
def get_income_statement():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': symbol.upper(),
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)  # Convert data to string
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol}],
        ids=[f"{symbol}_income_statement"]
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/news', methods=['GET'])
def get_stock_news():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': symbol.upper(),
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)  # Convert data to string
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol}],
        ids=[f"{symbol}_news"]
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/insider_transactions', methods=['GET'])
def get_insider_transactions():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'INSIDER_TRANSACTIONS',
        'symbol': symbol.upper(),
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)  # Convert data to string
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol}],
        ids=[f"{symbol}_insider_transactions"]
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/time_series', methods=['GET'])
def get_stock_time_series():
    symbol = request.args.get('symbol')
    time_series_function = request.args.get('function', 'TIME_SERIES_DAILY')
    outputsize = request.args.get('outputsize', 'compact')
    datatype = request.args.get('datatype', 'json')

    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    if time_series_function not in ['TIME_SERIES_DAILY', 'TIME_SERIES_WEEKLY', 'TIME_SERIES_MONTHLY']:
        return jsonify({'error': 'Invalid time series function.'}), 400

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': time_series_function,
        'symbol': symbol.upper(),
        'outputsize': outputsize,
        'datatype': datatype,
        'apikey': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol, 'function': time_series_function, 'outputsize': outputsize}],
        ids=[f"{symbol}_{time_series_function}_{outputsize}"]
    )

    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    return jsonify(data)

@cache.cached()
@app.route('/stocks/daily', methods=['GET'])
def get_stock_daily():
    symbol = request.args.get('symbol')
    outputsize = request.args.get('outputsize', 'compact')
    datatype = request.args.get('datatype', 'json')

    # Validate required parameters
    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    # Prepare the API request to Alpha Vantage
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol.upper(),
        'outputsize': outputsize,
        'datatype': datatype,
        'apikey': API_KEY
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check for request errors
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Generate embedding
    data_text = json.dumps(data)
    embedding = generate_embedding(data_text)

    # Store in Chroma
    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[{'symbol': symbol, 'outputsize': outputsize}],
        ids=[f"{symbol}_daily_{outputsize}"]
    )

    # Check for API errors in the response
    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    # Return the fetched data as JSON
    return jsonify(data)

# -------------- Top Stocks & Chat Routes --------------

@cache.cached()
@app.route('/stocks/top_movers', methods=['GET'])
def get_top_movers():
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TOP_GAINERS_LOSERS',
        'apikey': API_KEY
    }

    app.logger.debug('Requesting Alpha Vantage API: %s', url)
    app.logger.debug('Parameters: %s', params)

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    app.logger.debug('Alpha Vantage Response Status: %s', response.status_code)
    app.logger.debug('Alpha Vantage Response Body: %s', response.text)

    # Top gainers, Top losers, and Top traders
    if 'top_gainers' in data and 'top_losers' in data and 'most_actively_traded' in data:
        top_gainers = data['top_gainers'][:10]
        top_losers = data['top_losers'][:10]
        most_active = data['most_actively_traded'][:10]

        combined_data = {
            'top_gainers': top_gainers,
            'top_losers': top_losers,
            'most_active': most_active
        }

        # Generate embedding
        data_text = json.dumps(combined_data)
        embedding = generate_embedding(data_text)

        # Store in ChromaDB (existing functionality)
        collection.add(
            embeddings=[embedding],
            documents=[data_text],
            metadatas=[{'type': 'top_movers'}],
            ids=['top_movers']
        )

        # Prepare the item to insert into Cosmos DB
        item = {
            'id': 'top_movers_' + datetime.datetime.utcnow().isoformat(),
            'partitionKey': 'top_movers',
            'data': combined_data,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }

        # Insert the item into Cosmos DB
        try:
            container.create_item(body=item)
        except exceptions.CosmosHttpResponseError as e:
            app.logger.error('An error occurred while inserting item into Cosmos DB: %s', str(e))

        return jsonify(combined_data)
    else:
        return jsonify({'error': 'No data found for top gainers, losers, or most active.'}), 400

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/query', methods=['POST'])
def query_data():
    data = request.get_json()
    query_text = data.get('query')

    if not query_text:
        return jsonify({'error': 'Please provide a query in the request body.'}), 400

    # Generate embedding for the query
    query_embedding = generate_embedding(query_text)

    # Query the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,  # Number of results to return
        include=['documents', 'metadatas']  # Include documents and metadata in the response
    )

    return jsonify(results)

# -------------- ML Processing Routes --------------

#Re-Ranking news articles
@app.route('/api/rank-news', methods=['POST'])
def rank_news():
    data = request.get_json()
    news_articles = data.get('newsArticles', [])

    if not news_articles:
        return jsonify({'error': 'No news articles provided.'}), 400

    # Perform ranking
    ranked_articles = rank_news_by_impact(news_articles)

    return jsonify(ranked_articles), 200

def rank_news_by_impact(news_articles):
    # Define weights for different factors
    SENTIMENT_WEIGHT = 0.5
    RECENCY_WEIGHT = 0.3
    SOURCE_WEIGHT = 0.2

    # Current date for recency calculation
    current_time = datetime.datetime.utcnow()

    # Predefined credibility scores for sources (example values)
    source_credibility = {
        'Reuters': 1.0,
        'Bloomberg': 0.9,
        'Wall Street Journal': 0.9,
        'CNBC': 0.8,
        'Yahoo Finance': 0.7,
        'Motley Fool': 0.6,
        'Seeking Alpha': 0.6,
        'Benzinga': 0.5,
        # Add more sources as needed
    }

    ranked_articles = []

    for idx, article in enumerate(news_articles):
        try:
            # Sentiment score: Convert to absolute value to capture extreme sentiments
            sentiment_score = abs(float(article.get('overall_sentiment_score', 0)))

            # Recency score: Inverse of the time difference in hours
            time_published = article.get('time_published')
            # Parse time_published, expected format: '20241023T224500'
            try:
                article_time = datetime.datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                time_diff = (current_time - article_time).total_seconds() / 3600  # Time difference in hours
                recency_score = 1 / (1 + time_diff)
            except ValueError as ve:
                app.logger.warning(f"Article {idx} has invalid time_published format: {time_published}. Setting recency_score to 0.")
                recency_score = 0  # If parsing fails, set recency to 0

            # Source credibility score
            source = article.get('source', '').strip()
            credibility_score = source_credibility.get(source, 0.5)  # Default credibility is 0.5

            # Calculate overall impact score
            impact_score = (SENTIMENT_WEIGHT * sentiment_score +
                            RECENCY_WEIGHT * recency_score +
                            SOURCE_WEIGHT * credibility_score)

            # Add impact_score to the article
            ranked_article = article.copy()
            ranked_article['impact_score'] = impact_score

            ranked_articles.append(ranked_article)

        except Exception as e:
            app.logger.error(f"Error processing article {idx}: {e}")
            continue  # Skip this article and proceed with others

    # Sort articles by impact_score in descending order
    ranked_articles = sorted(ranked_articles, key=lambda x: x.get('impact_score', 0), reverse=True)

    return ranked_articles


if __name__ == '__main__':
    app.run(debug=True)