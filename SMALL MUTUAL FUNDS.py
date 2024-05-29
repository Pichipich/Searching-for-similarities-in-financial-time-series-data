

import yfinance as yf
import pandas as pd
import sqlite3

symbols = [
    "WWWFX", "KINCX", "KINAX", "KMKNX", "KMKCX", "KMKAX", "KMKYX", "KNPAX", "KNPYX", "KNPCX",
    "WWNPX", "KSCYX", "KSOAX", "KSCOX", "KSOCX", "LSHEX", "LSHAX", "LSHUX", "LSHCX", "ENPIX",
    "ENPSX", "FNRCX", "FSENX", "FIKAX", "FANAX", "FANIX", "FAGNX", "NEEIX", "NEEGX", "FIJGX",
    "FSTCX", "FTUTX", "FTUCX", "FTUAX", "FTUIX", "RCMFX", "FNARX", "HICGX", "HFCGX",
    "RYPMX", "RYMNX", "RYMPX", "RYZCX", "FUGAX", "FCGCX", "FIQRX", "FFGTX", "FFGAX", "FFGIX",
    "FFGCX", "FIKIX", "FUGIX", "FSUTX", "FAUFX", "FUGCX", "BIVIX",  "BIVRX", "NEAIX",
    "FSLBX", "NEAGX", "QLEIX", "QLERX", "FACVX", "FTCVX", "FIQVX", "FICVX", "FSPCX", "RMLPX",
    "FCCVX", "FCVSX", "EAFVX", "EIFVX", "DGIFX", "AUERX", "COAGX", "TAVZX", "TAVFX", "TVFVX",
    "ECFVX", "SGGDX", "EICVX", "EICIX", "MBXAX", "UBVVX", "UBVAX", "UBVFX", "MBXIX", "FEURX",
    "UBVRX", "UBVTX", "UBVUX", "UBVSX", "DHTAX", "UBVLX", "UBVCX",  "MBXCX", "DHTYX","HWSCX",
    "HWSIX", "EIPIX", "HWSAX"
]



conn = sqlite3.connect('mutual_small_data.db')

symbol_count = len(symbols)
print("Total number of symbols:", symbol_count)




desired_row_count = 209
# Define the start and end dates for your data retrieval
start_date = '2019-01-01'
end_date = '2022-12-31'

for symbol in symbols:
    # Download data from Yahoo Finance
    df = yf.download(symbol, start=start_date, end=end_date, interval='1wk')

    # Check if the DataFrame has the desired number of rows
    if len(df) != desired_row_count:
        print(f"{symbol} does not have exactly {desired_row_count} rows, it has {len(df)} rows. Skipping.")
        continue  # Skip this iteration and don't download or write to the database

    # If the check passes, proceed with data manipulation
    df['Symbol'] = symbol  # Add the 'Symbol' column to the DataFrame
    df['Date'] = df.index  # Add a column with the dates
    df.set_index('Date', inplace=True)  # Set 'Date' as the index

    # Write the DataFrame to SQLite database
    df.to_sql(symbol, conn, index=True, if_exists='replace')

# Commit and close the connection
conn.commit()
conn.close()


company_mapping = {
    "WWWFX": "Kinetics",
    "KINCX": "Kinetics", "KINAX": "Kinetics", "KMKNX": "Kinetics", "KMKCX": "Kinetics", "KMKAX": "Kinetics", "KMKYX": "Kinetics","KNPAX": "Kinetics","KNPYX": "Kinetics", "KNPCX": "Kinetics", "WWNPX": "Kinetics","KSCYX": "Kinetics","KSOAX": "Kinetics",
    "KSCOX": "Kinetics", "KSOCX": "Kinetics", "LSHEX": "Kinetics", "LSHAX": "Kinetics", "LSHUX": "Kinetics", "LSHCX": "Kinetics", "ENPIX": "ProFunds", "ENPSX": "ProFunds",    "FNRCX": "Fidelity",
    "FSENX": "Fidelity",  "FIKAX": "Fidelity",  "FANAX": "Fidelity",   "FANIX": "Fidelity", "FAGNX": "Fidelity",
    "NEEIX": "Needham","NEEGX": "Needham", "FIJGX": "Fidelity", "FSTCX": "Fidelity", "FTUTX": "Fidelity", "FTUCX": "Fidelity",
    "FTUAX": "Fidelity", "FTUIX": "Fidelity","RCMFX": "Schwartz", "FNARX": "Fidelity",
    "FMEIX": "Fidelity",  "HICGX": "Hennessy",  "HFCGX": "Hennessy",  "RYPMX": "Rydex",  "RYMNX": "Rydex",   "RYMPX": "Rydex",   "RYZCX": "Rydex",   "FUGAX": "Fidelity",   "FCGCX": "Fidelity",  "FIQRX": "Fidelity", "FFGTX": "Fidelity", "FFGAX": "Fidelity",
    "FFGIX": "Fidelity", "FFGCX": "Fidelity", "FIKIX": "Fidelity","FUGIX": "Fidelity", "FSUTX": "Fidelity",
    "FAUFX": "Fidelity",   "FUGCX": "Fidelity",   "BIVIX": "Invenomic",   "BIVSX": "Invenomic", "BIVRX": "Invenomic","NEAIX": "Needham",
    "FSLBX": "Fidelity","NEAGX": "Needham", "QLEIX": "AQR Long-Short Equity ", "QLERX": "AQR Long-Short Equity ",  "FACVX": "Fidelity",
    "FTCVX": "Fidelity",  "FIQVX": "Fidelity",  "FICVX": "Fidelity", "FSPCX": "Fidelity",  "RMLPX": "Two Roads Shared Trust",   "FCCVX": "Fidelity",   "FCVSX": "Fidelity",  "EAFVX": "Eaton Vance",   "EIFVX": "Eaton Vance", "DGIFX": "Disciplined Growth Investors",
    "AUERX": "Auer Growth",   "COAGX": "Caldwell & Orkin - Gator Capital L/S Fd",
    "TAVZX": "Third Avenue Value","TAVFX": "Third Avenue Value",  "TVFVX": "Third Avenue Value",  "ECFVX": "Eaton Vance",  "SGGDX": "First Eagle Gold",  "EICVX": "EIC", "EICIX": "EIC",
    "MBXAX": "Catalyst/Millburn Hedge Strategy Fund",  "UBVVX": "Undiscovered Managers",  "UBVAX": "Undiscovered Managers",  "UBVFX": "Undiscovered Managers",
    "MBXIX": "Catalyst/Millburn Hedge Strategy Fund",  "FEURX": "First Eagle Gold",
    "UBVRX": "Undiscovered Managers", "UBVTX": "Undiscovered Managers", "UBVUX": "Undiscovered Managers", "UBVSX": "Undiscovered Managers",  "DHTAX": "Diamond Hill Select Fund",
    "UBVLX": "Undiscovered Managers","UBVCX": "Undiscovered Managers", "MBXFX": "Catalyst/Millburn Hedge Strategy Fund", "MBXCX": "Catalyst/Millburn Hedge Strategy Fund", "DHTYX": "Diamond Hill Select Fund"
}