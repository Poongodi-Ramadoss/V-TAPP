# app.py
from flask import Flask
from flask_cors import CORS  # Import CORS
from controllers.recommendation_controller import recommendation_bp
import os
from controllers.db import get_db_connection

app = Flask(__name__)
CORS(app)

# Register the Blueprints
app.register_blueprint(recommendation_bp)

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_RUN_HOST', '0.0.0.0'), port=int(os.getenv('FLASK_RUN_PORT', 6000)), debug=True)