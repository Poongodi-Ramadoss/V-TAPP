from flask import Blueprint
import psycopg2  # PostgreSQL database adapter
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

recommendation_bp = Blueprint('recommendations', __name__)
def get_db_connection():
    """Establish and return a database connection using environment variables."""
    try:
        database_url = os.getenv('DB_URL')
        supabase: Client = create_client('https://qlqsmadmjmmcipubfzwo.supabase.co',os.getenv('DB_KEY'))
        # response = supabase.table("wishlists").select("*").execute()
        return supabase
    except Exception as e:
        # recommendation_bp.logger.error(f"Error connecting to the database: {str(e)}")
        raise