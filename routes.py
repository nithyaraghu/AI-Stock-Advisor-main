# routes.py

from flask import Blueprint, request, jsonify, current_app
import requests
import datetime
import logging
import azure.cosmos.exceptions as exceptions
import json

from utils import generate_embedding
from db import collection, container
from config import API_KEY
from flask_caching import Cache

routes_bp = Blueprint('routes', __name__)

# Initialize cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

@routes_bp.route('/stocks/quote', methods=['GET'])
@cache.cached()
def get_stock_quote():
    symbol = request.args.get('symbol')
    return fetch_and_store_stock_data('GLOBAL_QUOTE', symbol)

@routes_bp.route('/stocks/overview', methods=['GET'])
@cache.cached()
def get_stock_overview():
    symbol = request.args.get('symbol')
    chroma_id = f"{symbol}_overview"
    return fetch_and_store_stock_data('OVERVIEW', symbol, chroma_id=chroma_id)

@routes_bp.route('/stocks/income_statement', methods=['GET'])
@cache.cached()
def get_income_statement():
    symbol = request.args.get('symbol')
    chroma_id = f"{symbol}_income_statement"
    return fetch_and_store_stock_data('INCOME_STATEMENT', symbol, chroma_id=chroma_id)

@routes_bp.route('/stocks/news', methods=['GET'])
@cache.cached()
def get_stock_news():
    symbol = request.args.get('symbol')
    chroma_id = f"{symbol}_news"
    return fetch_and_store_stock_data('NEWS_SENTIMENT', symbol, chroma_id=chroma_id)

@routes_bp.route('/stocks/insider_transactions', methods=['GET'])
@cache.cached()
def get_insider_transactions():
    symbol = request.args.get('symbol')
    chroma_id = f"{symbol}_insider_transactions"
    return fetch_and_store_stock_data('INSIDER_TRANSACTIONS', symbol, chroma_id=chroma_id)

@routes_bp.route('/stocks/time_series', methods=['GET'])
@cache.cached()
def get_stock_time_series():
    symbol = request.args.get('symbol')
    time_series_function = request.args.get('function', 'TIME_SERIES_DAILY')
    outputsize = request.args.get('outputsize', 'compact')
    datatype = request.args.get('datatype', 'json')

    params = {
        'outputsize': outputsize,
        'datatype': datatype
    }

    chroma_id = f"{symbol}_{time_series_function}_{outputsize}"
    metadata = {
        'function': time_series_function,
        'outputsize': outputsize,
    }

    return fetch_and_store_stock_data(time_series_function, symbol, params=params, chroma_id=chroma_id, metadata=metadata)

@routes_bp.route('/stocks/daily', methods=['GET'])
@cache.cached()
def get_stock_daily():
    symbol = request.args.get('symbol')
    outputsize = request.args.get('outputsize', 'compact')
    datatype = request.args.get('datatype', 'json')

    params = {
        'outputsize': outputsize,
        'datatype': datatype
    }

    chroma_id = f"{symbol}_daily_{outputsize}"
    metadata = {
        'outputsize': outputsize,
    }

    return fetch_and_store_stock_data('TIME_SERIES_DAILY', symbol, params=params, chroma_id=chroma_id, metadata=metadata)

# Helper function to fetch and store stock data
def fetch_and_store_stock_data(function_name, symbol, params={}, chroma_id=None, metadata={}):
    if not symbol:
        return jsonify({'error': 'Please provide the stock symbol as a parameter.'}), 400

    url = 'https://www.alphavantage.co/query'
    default_params = {
        'function': function_name,
        'symbol': symbol.upper(),
        'apikey': API_KEY
    }
    default_params.update(params)

    response = requests.get(url, params=default_params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from Alpha Vantage API.'}), 500

    data = response.json()

    # Check for API errors
    if 'Error Message' in data or 'Note' in data:
        return jsonify({'error': data.get('Error Message') or data.get('Note', 'API call limit reached.')}), 400

    # Generate embedding
    data_text = json.dumps(data)
    embedding = generate_embedding(data_text)

    # Store in ChromaDB
    if not chroma_id:
        chroma_id = symbol

    metadata['symbol'] = symbol

    collection.add(
        embeddings=[embedding],
        documents=[data_text],
        metadatas=[metadata],
        ids=[chroma_id]
    )

    return jsonify(data)
