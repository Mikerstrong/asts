# Stock Data Fetcher

This project is designed to fetch stock data every hour while ensuring that the stock market is open. It utilizes a scheduling mechanism to automate the data retrieval process.

## Project Structure

```
stock-data-fetcher
├── src
│   ├── main.py                # Entry point of the application
│   ├── services
│   │   └── stock_service.py   # Contains StockService class for fetching stock data
│   ├── utils
│   │   └── time_utils.py      # Utility functions for time management
│   └── types
│       └── stock_data.py      # Defines the StockData class
├── requirements.txt           # Lists project dependencies
├── .gitignore                 # Specifies files to ignore in Git
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-data-fetcher
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the following command:
```
python src/main.py
```

The application will check if the stock market is open and fetch the latest stock data every hour.

## Dependencies

- `requests`: For making HTTP requests to fetch stock data.
- `schedule`: For scheduling the hourly data fetching task.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.