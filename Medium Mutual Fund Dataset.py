import yfinance as yf
import pandas as pd
import sqlite3


symbols = [
    "RYPMX", "RYMPX", "RYMNX", "RYZCX", "ENPIX", "ENPSX", "FNARX", "JMCGX", "JMIGX",
    "FFGIX", "FFGCX", "FCGCX", "FIQRX", "FFGTX", "FFGAX", "SGGDX", "FEGIX", "KINCX",
    "WWWFX", "FEGOX", "KINAX", "FEURX", "KMKNX", "KMKCX",   "KMKYX",
    "GOFIX", "KMKAX", "GOVIX",  "FAGNX", "FIKAX", "FSENX", "FANIX", "FANAX",
    "KNPAX", "FNRCX", "KNPYX", "WWNPX", "KNPCX", "FKRCX", "FGPMX", "FGADX", "BIPIX",
    "BIPSX", "LSHEX", "KSCYX", "KSOCX", "KSCOX", "KSOAX", "LSHAX", "LSHUX", "LSHCX",
    "COBYX", "HICGX", "HFCGX", "MCMVX", "FGFLX", "FGFRX", "FGRSX", "FGFAX", "MOWNX",
    "MOWIX", "FGFCX", "RCMFX", "FSHOX", "TVFVX", "TAVZX", "TAVFX", "RMLPX", "TCMSX",
     "SVFAX", "SVFKX", "SVFDX", "SVFFX", "SMVLX", "SVFYX", "BIVRX",
    "HWAAX", "SSSIX", "HWAIX",  "BIVIX", "HWACX", "SSSFX", "UMPSX",
    "UMPIX", "FVIFX", "BGRSX",  "FTVFX", "HIMDX", "BGLSX", "CSERX", "FAVFX", "HFMDX", "SLVRX",
    "CPLSX", "CSRYX", "FVLZX", "SVLCX", "FDVLX", "FDVLX", "AUERX","FVLKX", "SLVAX", "CSVZX",
    "SLVIX", "CPCLX", "JORFX", "FSRPX", "YAFIX", "JORCX",
    "MBXIX",  "CPLIX", "MBXCX", "TGIRX", "CSVAX", "CSVFX", "THVRX", "THGCX", "CGOLX",
    "TGVRX", "JANRX", "JSLNX", "JORNX", "JORAX", "FUGAX", "JORIX", "YACKX", "YAFFX", "SNOIX",
     "BUFOX", "FUGCX",  "FIKIX", "FUGIX", "TIVRX", "TGVIX", "FSUTX",
    "SNOCX", "FAUFX", "HWLCX", "CSGRX", "TGVAX", "CADPX", "HWLAX", "FMDCX", "CLSYX", "SNOAX",
    "DSCPX", "HWLIX", "CSRCX", "JORRX", "HWCIX", "HULEX", "HULIX", "HWSAX", "FSTRX", "QRLVX",
    "FMSTX", "QCLVX", "FSTKX", "FSTLX", "HWSCX", "DHLAX", "DHLRX", "DHLYX", "HWCAX",
    "HWSIX", "HWCCX", "FMCRX", "FMCLX", "BLUEX", "TBGVX", "NECOX", "CSVRX", "NEOYX", "CSMIX",
    "TFIFX", "NOANX", "PRISX", "CVVRX", "CUURX", "CSCZX", "CSVYX", "FIJCX", "FSPCX", "FDIGX",
    "FDTGX", "FDFAX","TBWIX", "DODWX", "MRFOX", "FCVTX",  "FSLBX", "EIPFX", "FEVCX", "WBSNX", "CMIDX",
    "NEFJX", "THOIX", "FCVCX", "FCVAX", "IMIDX"

]

conn = sqlite3.connect('mutual_data.db')

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
    "RYPMX": "Rydex Precious Metals", "RYMNX": "Rydex Precious Metals", "RYMPX": "Rydex Precious Metals",
    "RYZCX": "Rydex Precious Metals", "FUGAX": "Fidelity", "FCGCX": "Fidelity", "FIQRX": "Fidelity",
    "FFGTX": "Fidelity", "FFGAX": "Fidelity", "FFGIX": "Fidelity", "FFGCX": "Fidelity", "FIKIX": "Fidelity",
    "FUGIX": "Fidelity", "FSUTX": "Fidelity", "FAUFX": "Fidelity", "FUGCX": "Fidelity", "BIVIX": "Invenomic Fund",
    "BIVSX": "Invenomic Fund", "BIVRX": "Invenomic Fund", "NEAIX": "Needham", "FSLBX": "Fidelity",
    "NEAGX": "Needham Aggressive Growth Retail", "QLEIX": "AQR Long-Short Equity", "QLERX": "AQR Long-Short Equity",
    "FACVX": "Fidelity", "FTCVX": "Fidelity", "FIQVX": "Fidelity", "FICVX": "Fidelity", "FSPCX": "Fidelity",
    "RMLPX": "Two Roads Shared Trust - Recurrent MLP & Infrastructure Fund", "FCCVX": "Fidelity", "FCVSX": "Fidelity",
    "EAFVX": "Eaton Vance Focused Value", "EIFVX": "Eaton Vance Focused Value", "DGIFX": "Disciplined Growth Investors",
    "AUERX": "Auer Growth", "COAGX": "Caldwell & Orkin - Gator Capital L/S Fd", "TAVZX": "Third Avenue Value",
    "TAVFX": "Third Avenue Value", "TVFVX": "Third Avenue Value", "ECFVX": "Eaton Vance Focused Value Opportunities Fund",
    "SGGDX": "First Eagle", "EICVX": "EIC Value A", "EICIX": "EIC Value Institutional",
    "MBXAX": "Mutual Fund Series Trust - Catalyst/Millburn Hedge Strategy Fund", "UBVVX": "Undiscovered Managers Behavioral Value Fund",
    "UBVAX": "Undiscovered Managers Behavioral Value Fund", "UBVFX": "Undiscovered Managers Behavioral Value Fund",
    "MBXIX": "Mutual Fund Series Trust - Catalyst/Millburn Hedge Strategy Fund", "FEURX": "First Eagle",
    "MBXCX": "Mutual Fund Series Trust - Catalyst/Millburn Hedge Strategy Fund", "TGIRX": "Thornburg International",
    "CSVAX": "Columbia", "CSVFX": "Columbia", "THVRX": "Thornburg International",
    "THGCX": "Thornburg International", "CGOLX": "Columbia", "TGVRX": "Thornburg International",
    "JANRX": "Janus Henderson", "JSLNX": "Janus Henderson", "JORNX": "Janus Henderson",
    "JORAX": "Janus Henderson", "FUGAX ": "Fidelity Advisor Utilities", "JORIX": "Janus Henderson",
    "YACKX": "AMG", "YAFFX": "AMG", "SNOIX": "Easterly Snow Long/Short Opportunity",
    "BUFOX": "Buffalo", "FUGCX ": "Fidelity", "FIKIX ": "Fidelity", "FUGIX ": "Fidelity",
    "TIVRX": "Thornburg International", "TGVIX": "Thornburg International", "FSUTX ": "Fidelity",
    "SNOCX": "Easterly Snow Long/Short Opportunity", "FAUFX ": "Fidelity", "HWLCX": "Hotchkis & Wiley",
    "CSGRX": "Columbia", "TGVAX": "Thornburg International", "CADPX": "Columbia",
    "HWLAX": "Hotchkis & Wiley", "FMDCX": "Federated Hermes", "CLSYX": "Columbia",
    "SNOAX": "Easterly Snow Long/Short Opportunity", "DSCPX": "Davenport", "HWLIX": "Hotchkis & Wiley",
    "CSRCX": "Columbia", "JORRX": "Janus Henderson", "HWCIX": "Hotchkis & Wiley", "HULEX": "Huber Select",
    "HULIX": "Huber Select", "HWSAX": "Hotchkis & Wiley", "FSTRX": "Federated Hermes", "QRLVX": "Federated Hermes",
    "FMSTX": "Federated Hermes", "QCLVX": "Federated Hermes", "FSTKX": "Federated Hermes", "FSTLX": "Federated Hermes",
    "HWSCX": "Hotchkis & Wiley", "DHLAX": "Diamond Hill", "DHLRX": "Diamond Hill", "DHLYX": "Diamond Hill",
    "HWCAX": "Hotchkis & Wiley", "HWSIX": "Hotchkis & Wiley", "HWCCX": "Hotchkis & Wiley", "FMCRX ": "Federated Hermes",
    "FMCLX": "Federated Hermes", "BLUEX": "AMG", "TBGVX": "Tweedy, Browne", "NECOX": "Natixis",
    "CSVRX": "Columbia", "NEOYX": "Natixis", "CSMIX": "Columbia", "TFIFX": "T. Rowe Price Financial Services",
    "NOANX": "Natixis", "PRISX": "T. Rowe Price Financial Services", "CVVRX": "Columbia", "CUURX": "Columbia",
    "CSCZX": "Columbia", "CSVYX": "Columbia", "FIJCX": "Fidelity", "FDIGX": "Fidelity",
    "FDTGX": "Fidelity", "FDFAX": "Fidelity", "TBWIX": "Thornburg International", "DODWX": "Dodge & Cox",
    "MRFOX": "Marshfield Concentrated Opportunity", "FCVTX": "Fidelity", "EIPFX": "EIP Growth and Income Investor",
    "FEVCX": "First Eagle", "WBSNX": "William Blair", "CMIDX": "Congress", "NEFJX": "Natixis",
    "THOIX": "Thornburg International", "FCVCX": "Fidelity", "FCVAX": "Fidelity", "IMIDX": "Congress"
}

