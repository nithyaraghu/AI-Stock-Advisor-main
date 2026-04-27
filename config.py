import os
from dotenv import load_dotenv

load_dotenv()

settings = {
    'database_url': os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:postgres@localhost:5432/stockadvisor'
    ),
}

API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'dev-jwt-secret-key')