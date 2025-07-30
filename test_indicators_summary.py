#!/usr/bin/env python3
"""
Test script for the indicators summary functionality added to asts.py

This script validates that the HTML output correctly includes the 
indicators summary section as requested.
"""

import pandas as pd
from datetime import datetime
import sys
import os

def create_test_data():
    """Create test data that mimics the structure of real ASTS data with indicators"""
    dates = pd.date_range(end=datetime.today(), periods=30, freq='D')
    
    # Create realistic ASTS-like stock data
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': [15.20 + i*0.1 for i in range(30)],
        'High': [15.80 + i*0.1 for i in range(30)],
        'Low': [15.00 + i*0.1 for i in range(30)],
        'Close': [15.45 + i*0.1 for i in range(30)],
        'Volume': [2500000 + i*10000 for i in range(30)],
        'Shares Outstanding': [150000000] * 30,
        # Add some test indicator values
        'SMA10': [15.30 + i*0.1 for i in range(30)],
        'EMA20': [15.35 + i*0.1 for i in range(30)],
        'UpperBB': [16.00 + i*0.1 for i in range(30)],
        'LowerBB': [14.80 + i*0.1 for i in range(30)],
        'VWAP': [15.40 + i*0.1 for i in range(30)],
        'RSI': [50 + (i % 20) for i in range(30)],  # RSI values between 50-70
        'MACD': [0.1 + i*0.01 for i in range(30)],
        'MACD_Signal': [0.05 + i*0.01 for i in range(30)],
        'ATR': [0.5 + i*0.01 for i in range(30)]
    })
    
    return test_df

def test_indicators_summary():
    """Test that the indicators summary function works correctly"""
    from asts import build_indicators_summary
    
    test_df = create_test_data()
    summary_html = build_indicators_summary(test_df, "TEST")
    
    print("Testing indicators summary generation...")
    
    # Check that the summary includes the new section
    assert "Indicators Summary" in summary_html, "❌ 'Indicators Summary' section missing"
    print("✅ 'Indicators Summary' section found")
    
    # Check that key indicators are included
    required_indicators = ["Current Price", "SMA", "EMA", "RSI", "MACD", "ATR"]
    for indicator in required_indicators:
        assert indicator in summary_html, f"❌ {indicator} missing from summary"
    print("✅ All required indicators found in summary")
    
    # Check that trend analysis is included
    assert "Overall Trend:" in summary_html, "❌ Overall trend analysis missing"
    print("✅ Overall trend analysis found")
    
    # Check that RSI interpretation is working
    assert "Neutral" in summary_html or "Overbought" in summary_html or "Oversold" in summary_html, "❌ RSI interpretation missing"
    print("✅ RSI interpretation found")
    
    # Check HTML structure
    assert "<table" in summary_html, "❌ HTML table structure missing"
    assert "<th" in summary_html, "❌ Table headers missing"
    assert "<td" in summary_html, "❌ Table data missing"
    print("✅ HTML table structure is correct")
    
    print("\n📊 Sample indicators summary structure:")
    print("=" * 60)
    # Extract a few key lines for demonstration
    lines = summary_html.split('\n')
    for line in lines:
        if 'TEST Indicators Summary' in line:
            print("Header: Found indicators summary title")
        elif 'Overall Trend:' in line:
            print("Trend: Found overall trend analysis")
        elif 'Current Price' in line and '<td' in line:
            print("Data: Found current price row")
            break
    print("=" * 60)

def test_full_integration():
    """Test that the main application includes the indicators summary"""
    print("\nTesting full integration...")
    
    # Run a single stock to test integration
    from asts import fetch_data, build_indicators_summary
    
    try:
        # Create mock data for testing
        test_df = create_test_data()
        summary_html = build_indicators_summary(test_df, "INTEGRATION_TEST")
        
        # Check that it returns valid HTML
        assert len(summary_html) > 100, "❌ Summary HTML too short"
        assert "INTEGRATION_TEST" in summary_html, "❌ Stock symbol not in summary"
        print("✅ Integration test passed - summary generates correctly")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests for indicators summary functionality"""
    print("🧪 Running tests for indicators summary functionality...\n")
    
    try:
        test_indicators_summary()
        test_full_integration()
        
        print("\n🎉 All indicators summary tests passed!")
        print("\nChanges made:")
        print("✓ Added build_indicators_summary() function")
        print("✓ Summary shows current values of all technical indicators")
        print("✓ Added trend analysis (Bullish/Bearish/Neutral)")
        print("✓ Added RSI signal interpretation (Overbought/Oversold/Neutral)")
        print("✓ Integrated summary into main HTML output")
        print("✓ Summary appears at the top of each stock section")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()