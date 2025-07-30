#!/usr/bin/env python3
"""
ASTS Dashboard Web Server
Serves the stock dashboard with automatic data updates
"""

import os
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timedelta
import schedule
import subprocess
import sys

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler for the ASTS dashboard"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/app", **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/asts.html'
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "timestamp": "' + 
                           datetime.now().isoformat().encode() + b'"}')
            return
        elif self.path == '/update':
            # Trigger dashboard update
            self.update_dashboard()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "update_triggered"}')
            return
        
        super().do_GET()
    
    def update_dashboard(self):
        """Update the dashboard by running the Python script"""
        try:
            subprocess.run([sys.executable, "asts.py"], check=True)
            print(f"Dashboard updated at {datetime.now()}")
        except subprocess.CalledProcessError as e:
            print(f"Error updating dashboard: {e}")

def is_market_open():
    """Check if the stock market is open (US Eastern Time)"""
    now = datetime.now()
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    
    # Market is closed on weekends
    if weekday >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET (simplified)
    hour = now.hour
    return 9 <= hour <= 16

def update_dashboard_job():
    """Job to update dashboard during market hours"""
    if is_market_open():
        print(f"Market is open, updating dashboard at {datetime.now()}")
        try:
            subprocess.run([sys.executable, "asts.py"], check=True)
            print("Dashboard updated successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error updating dashboard: {e}")
    else:
        print(f"Market is closed, skipping update at {datetime.now()}")

def schedule_updates():
    """Schedule dashboard updates every hour during market hours"""
    # Schedule updates every hour from 9 AM to 4 PM on weekdays
    for hour in range(9, 17):
        schedule.every().monday.at(f"{hour:02d}:00").do(update_dashboard_job)
        schedule.every().tuesday.at(f"{hour:02d}:00").do(update_dashboard_job)
        schedule.every().wednesday.at(f"{hour:02d}:00").do(update_dashboard_job)
        schedule.every().thursday.at(f"{hour:02d}:00").do(update_dashboard_job)
        schedule.every().friday.at(f"{hour:02d}:00").do(update_dashboard_job)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Run scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("Scheduler started for hourly market updates")

def main():
    """Main function to start the web server"""
    # Initial dashboard generation
    print("Generating initial dashboard...")
    try:
        subprocess.run([sys.executable, "asts.py"], check=True)
        print("Initial dashboard generated")
    except subprocess.CalledProcessError as e:
        print(f"Error generating initial dashboard: {e}")
    
    # Start scheduler
    schedule_updates()
    
    # Start web server
    port = int(os.environ.get('PORT', 8000))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    
    print(f"Starting ASTS Dashboard Server on port {port}")
    print(f"Dashboard will update hourly during market hours (9 AM - 4 PM ET)")
    print(f"Access the dashboard at: http://localhost:{port}")
    print(f"Health check at: http://localhost:{port}/health")
    print(f"Manual update at: http://localhost:{port}/update")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()
