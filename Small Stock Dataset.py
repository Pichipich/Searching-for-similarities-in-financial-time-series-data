import yfinance as yf
import sqlite3


symbols = ['AMAT', 'JPM', 'NVDA', 'GE', 'HD', 'INTC', 'QCOM', 'HON', 'WFC',
           'AAPL', 'MSFT', 'TXN', 'GOOG', 'USB', 'GOOGL', 'GILD', 'AMZN', 'PEP',
           'LLY', 'MNST', 'CDNS', 'META', 'MU', 'TSLA', 'HSBC',
           'T', 'CVS', 'BA', 'UNH', 'TSM', 'V', 'NVO', 'SHEL', 'BBY', 'WMT', 'XOM',
           'MA', 'JNJ', 'AVGO', 'PG', 'ORCL', 'BHP', 'AZN', 'ADBE', 'ASML', 'CVX',
           'MRK', 'COST', 'TM', 'ABBV', 'KO', 'NGG', 'CRM', 'BAC', 'LEN', 'ACN', 'MCD',
           'NVS', 'BIIB', 'NFLX', 'LIN', 'SAP', 'CSCO', 'AMD', 'TMO', 'PDD', 'BABA',
           'ABT', 'TMUS', 'NKE', 'TTE', 'TBC', 'CMCSA', 'DIS', 'PFE', 'DHR', 'VZ', 'TBB',
           'INTU', 'PHM', 'LYG', 'IBM', 'AMGN', 'PM', 'UNP', 'NOW', 'RYAAY', 'COP',
           'SPGI', 'TFC', 'MS', 'UPS', 'CAT', 'RY', 'AXP', 'UL', 'NEE', 'RTX',
           'LOW', 'SNY']


conn = sqlite3.connect('small_stock_dataset.db')
symbol_count = len(symbols)
print("Total number of symbols:", symbol_count)

desired_row_count = 209

# Iterate over each symbol
for symbol in symbols:
    # Download data from Yahoo Finance
    df = yf.download(symbol, start='2019-01-01', end='2022-12-31', interval='1wk')

    # Check if the DataFrame has exactly 209 rows
    if len(df) != desired_row_count:
        # If not, print the symbol and skip this iteration
        print(f"{symbol} does not have exactly 209 rows, it has {len(df)} rows. Skipping download.")
        continue

    # If the DataFrame has 209 rows, proceed
    # Add the 'Symbol' column to the DataFrame
    df['Symbol'] = symbol

    # Add a column with the dates
    df['Date'] = df.index

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Write the DataFrame to SQLite database
    df.to_sql(symbol, conn, if_exists='replace', index=True)

# Commit and close the connection
conn.commit()
conn.close()




sector_mapping = {
    "AMAT": "Technology", "JPM": "Finance", "NVDA": "Technology", "HD": "Consumer Discretionary",
    "INTC": "Technology", "QCOM": "Technology", "HON": "Industrials", "WFC": "Finance",
    "AAPL": "Technology", "MSFT": "Technology", "TXN": "Technology", "GOOG": "Technology",
    "USB": "Finance", "GOOGL": "Technology", "GILD": "Health Care", "AMZN": "Consumer Discretionary",
    "PEP": "Consumer Staples", "LLY": "Health Care", "MNST": "Consumer Staples", "CDNS": "Technology",
    "META": "Technology", "MU": "Technology", "TSLA": "Consumer Discretionary", "HSBC": "Finance",
    "T": "Telecommunications", "CVS": "Consumer Staples", "BA": "Industrials", "UNH": "Health Care",
    "TSM": "Technology", "V": "Consumer Discretionary", "NVO": "Health Care", "SHEL": "Energy",
    "BBY": "Consumer Discretionary", "WMT": "Consumer Discretionary", "XOM": "Energy",
    "MA": "Consumer Discretionary", "JNJ": "Health Care", "AVGO": "Technology", "PG": "Consumer Discretionary",
    "ORCL": "Technology", "BHP": "Basic Materials", "AZN": "Health Care", "ADBE": "Technology",
    "ASML": "Technology", "CVX": "Energy", "MRK": "Health Care", "COST": "Consumer Discretionary",
    "TM": "Consumer Discretionary", "ABBV": "Health Care", "KO": "Consumer Staples",
    "NGG": "Utilities", "CRM": "Technology", "BAC": "Finance", "LEN": "Consumer Discretionary",
    "ACN": "Technology", "MCD": "Consumer Discretionary", "NVS": "Health Care", "BIIB": "Health Care",
    "NFLX": "Consumer Discretionary", "LIN": "Industrials", "SAP": "Technology", "CSCO": "Technology",
    "AMD": "Technology", "TMO": "Health Care", "PDD": "Consumer Discretionary", "BABA": "Consumer Discretionary",
    "ABT": "Health Care", "TMUS": "Telecommunications", "NKE": "Consumer Discretionary",
    "TTE": "Energy", "TBC": "Telecommunications", "CMCSA": "Consumer Discretionary",
    "DIS": "Consumer Discretionary", "PFE": "Health Care", "DHR": "Health Care", "VZ": "Telecommunications",
    "TBB": "Telecommunications", "INTU": "Technology", "PHM": "Consumer Discretionary",
    "LYG": "Finance", "IBM": "Technology", "AMGN": "Health Care", "PM": "Consumer Staples",
    "UNP": "Industrials", "NOW": "Technology", "RYAAY": "Consumer Discretionary", "COP": "Energy",
    "SPGI": "Finance", "TFC": "Finance", "MS": "Finance", "UPS": "Industrials", "CAT": "Industrials",
    "RY": "Finance", "AXP": "Finance", "UL": "Consumer Staples", "NEE": "Utilities", "UBER": "Consumer Discretionary",
    "RTX": "Industrials", "LOW": "Consumer Discretionary", "SNY": "Health Care"
}
