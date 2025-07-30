#!/usr/bin/env python3
"""
Simple Flask app for ASTS Stock Dashboard
Can be deployed on Heroku, Railway, Render, or any Python hosting service
"""

from flask import Flask, render_template, jsonify, send_from_directory
import os
import sys
import schedule
import time
import threading
from datetime import datetime
import pytz
import logging

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our existing ASTS module
import asts

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_market_open():
    """Check if US stock market is open (9 AM - 4 PM ET, Mon-Fri)"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's during market hours (9 AM - 4 PM ET)
    market_open = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close

def update_data():
    """Update stock data and regenerate charts"""
    try:
        logger.info("Updating ASTS stock data...")
        
        # Fetch fresh data using the correct function name
        df = asts.fetch_data("ASTS")
        
        # Generate Plotly chart
        asts.generate_plotly_chart(df)
        
        # Generate Seaborn chart (need to check if this function exists)
        try:
            asts.generate_seaborn_chart(df)
        except AttributeError:
            logger.warning("generate_seaborn_chart function not found, skipping seaborn chart")
        
        logger.info("ASTS data updated successfully")
    except Exception as e:
        logger.error(f"Error updating ASTS data: {e}")

def schedule_updates():
    """Schedule automatic updates during market hours"""
    schedule.every().hour.do(lambda: update_data() if is_market_open() else None)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Start the scheduler in a background thread
scheduler_thread = threading.Thread(target=schedule_updates, daemon=True)
scheduler_thread.start()

@app.route('/')
def index():
    """Serve the main dashboard"""
    try:
        # Generate fresh charts on page load
        update_data()
        return render_template('asts.html')
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'market_open': is_market_open()
    })

@app.route('/update')
def manual_update():
    """Manual update endpoint"""
    try:
        update_data()
        return jsonify({
            'status': 'success',
            'message': 'Data updated successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Manual update failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (charts, etc.)"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployments) or default to 8000
    port = int(os.environ.get('PORT', 8000))
    
    # Generate initial charts
    logger.info("Generating initial charts...")
    update_data()
    
    # Run the Flask app
    logger.info(f"Starting ASTS Dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
