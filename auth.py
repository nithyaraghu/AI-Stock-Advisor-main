# auth.py

from flask import Blueprint, request, jsonify, session
import bcrypt
from db import container
import azure.cosmos.exceptions as exceptions

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['POST'])
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
        "investmentGoal": investment_goal,
        "riskAppetite": risk_appetite,
        "timeHorizon": time_horizon
    }

    try:
        # Insert new user into Cosmos DB container
        container.create_item(body=new_user)
        return jsonify({"message": "User signed up successfully"}), 201
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"message": "Error creating user", "error": str(e)}), 500

# Login Route
@auth_bp.route('/login', methods=['POST'])
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

    # Store user information in session
    session['user_id'] = user['email']
    session['username'] = user['username']

    return jsonify({"message": "Login successful", "username": user['username']}), 200

# Logout Route
@auth_bp.route('/logout', methods=['POST'])
def logout():
    # Clear session
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200
