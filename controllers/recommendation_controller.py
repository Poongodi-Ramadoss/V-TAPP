from flask import Blueprint, request, jsonify
from flask_cors import CORS
import pandas as pd
from ml.recommendation1 import recommend_books_based_on_wishlist # Import your recommendation function
from controllers.db import get_db_connection

recommendation_bp = Blueprint('recommendations', __name__)

@recommendation_bp.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    # data = request.json
    # user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        # Connect to the database
        conn = get_db_connection()
        
        # Load data from the database
        # wishlist_df = pd.read_sql_query('SELECT * FROM wishlist', conn)
        wishlist_df = conn.table('wishlists').select('*').execute()
        ratings_df =conn.table('books_ratings').select('*').execute()
        books_df = conn.table('books').select('*').execute()
        
        print('Loading recommandation for user {user_id}')
        # Get recommendations
        recommended_books = recommend_books_based_on_wishlist(user_id, wishlist_df, ratings_df, books_df)
        print(recommended_books)
        print(jsonify(recommended_books.to_dict(orient='records')))
        return jsonify(recommended_books.to_dict(orient='records'))

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
        return jsonify({"error": str(e)}), 500