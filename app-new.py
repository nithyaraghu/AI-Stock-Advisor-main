from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_caching import Cache
import logging
import os

from auth import auth_bp
from routes import routes_bp
from db import container, collection
from config import settings

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# Initialize cache
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(routes_bp)

# Before and after request logging
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

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
