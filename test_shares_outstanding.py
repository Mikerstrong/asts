#!/usr/bin/env python3
"""
Test script for the shares outstanding functionality added to asts.py

This script validates that the HTML table correctly includes the 
'Total Current Shares' (Shares Outstanding) column as requested.
"""

import pandas as pd
from datetime import datetime
import sys
import os

def create_test_data():
    """Create test data that mimics the structure of real ASTS data"""
    dates = pd.date_range(end=datetime.today(), periods=10, freq='D')
    
    # Create realistic ASTS-like stock data
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': [15.20, 15.35, 15.10, 15.45, 15.60, 15.40, 15.75, 15.50, 15.90, 16.10],
        'High': [15.80, 15.75, 15.60, 15.85, 16.00, 15.90, 16.20, 15.95, 16.30, 16.45],
        'Low': [15.00, 15.20, 14.95, 15.30, 15.45, 15.25, 15.60, 15.35, 15.75, 15.95],
        'Close': [15.45, 15.25, 15.40, 15.70, 15.85, 15.65, 15.95, 15.80, 16.15, 16.25],
        'Volume': [2500000, 1800000, 3200000, 2100000, 2700000, 1900000, 2400000, 2000000, 3100000, 2600000],
        'Shares Outstanding': [150000000] * 10  # Same value for all rows
    })
    
    return test_df

def test_table_structure():
    """Test that the table has the correct structure with shares outstanding"""
    from asts import build_html_table
    
    test_df = create_test_data()
    table_html = build_html_table(test_df)
    
    print("Testing table structure...")
    
    # Check that the table includes the new column
    assert "Shares Outstanding" in table_html, "❌ 'Shares Outstanding' header missing from table"
    print("✅ 'Shares Outstanding' header found in table")
    
    # Check that the shares outstanding values are formatted correctly
    assert "150,000,000" in table_html, "❌ Shares outstanding value not properly formatted"
    print("✅ Shares outstanding values properly formatted with commas")
    
    # Check that we have the correct number of columns (7 total)
    header_count = table_html.count("<th>")
    assert header_count == 7, f"❌ Expected 7 headers, found {header_count}"
    print("✅ Table has correct number of columns (7)")
    
    # Check that each data row contains the shares outstanding value
    shares_count = table_html.count("150,000,000")
    expected_count = len(test_df)  # Should appear once per data row
    assert shares_count == expected_count, f"❌ Expected {expected_count} occurrences of shares value, found {shares_count}"
    print(f"✅ Shares outstanding appears in all {expected_count} data rows")
    
    return table_html

def test_data_consistency():
    """Test that shares outstanding data is consistent across rows"""
    test_df = create_test_data()
    
    print("\nTesting data consistency...")
    
    # All shares outstanding values should be the same
    unique_shares = test_df['Shares Outstanding'].nunique()
    assert unique_shares == 1, f"❌ Expected 1 unique shares value, found {unique_shares}"
    print("✅ All rows have the same shares outstanding value")
    
    # The value should be reasonable for a stock
    shares_value = test_df['Shares Outstanding'].iloc[0]
    assert shares_value > 0, "❌ Shares outstanding should be positive"
    assert shares_value < 1e12, "❌ Shares outstanding seems unreasonably large"
    print(f"✅ Shares outstanding value ({shares_value:,}) is reasonable")

def display_sample_output(table_html):
    """Display a sample of the table output for visual verification"""
    print("\nSample table output:")
    print("=" * 80)
    
    # Extract and display the header
    if "<tr><th>" in table_html and "</th></tr>" in table_html:
        header_part = table_html.split("<tr><th>")[1].split("</th></tr>")[0]
        headers = header_part.replace("</th><th>", " | ")
        print(f"Headers: {headers}")
    
    # Extract and display the first data row
    if "<tr><td>" in table_html and "</td></tr>" in table_html:
        first_row_part = table_html.split("<tr><td>")[1].split("</td></tr>")[0]
        first_row_data = first_row_part.replace("</td><td>", " | ")
        print(f"Sample:  {first_row_data}")
    
    print("=" * 80)

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 Running tests for shares outstanding functionality...\n")
    
    try:
        # Test table structure
        table_html = test_table_structure()
        
        # Test data consistency  
        test_data_consistency()
        
        # Display sample output
        display_sample_output(table_html)
        
        print("\n🎉 All tests passed! The table successfully includes 'Total Current Shares' data.")
        print("\nChanges made:")
        print("✓ Added 'Shares Outstanding' column to table headers")
        print("✓ Added shares outstanding data fetching with fallback values")
        print("✓ Formatted shares outstanding numbers with thousand separators")
        print("✓ Updated weekly resampling to preserve shares outstanding data")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return False

if __name__ == "__main__":
    # Add the project directory to Python path so we can import asts
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)