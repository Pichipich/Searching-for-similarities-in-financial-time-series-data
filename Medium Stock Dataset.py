import yfinance as yf
import sqlite3

symbols = [
    "FLT", "CNP", "MKC", "VIV", "PFG", "DECK", "ILMN",  "TSN", "WBA",
    "BLDR", "BMRN", "CHKP", "EXPD", "PHG", "CLX", "AKAM", "ZTO", "FITBI", "AXON",
    "SIRI", "TYL", "TVE", "EG", "VRT", "HRL",  "FDS",
    "AGNCN", "ATO", "YUMC", "NOK",  "HBAN", "WAT", "AER", "LPLA", "CMS",
    "NTRS", "BAH",  "WLK", "HOLX", "COO", "FSLR", "ALGN",
    "SUI", "LUV", "CINF", "OMC", "STX", "J", "HUBB",  "SWKS",
    "RF", "EQT", "ENTG", "L",  "MGA","WSO", "AVY",
    "SLMBP", "RS",  "BG", "IEX", "CE", "KB", "DGX", "WDC",
    "LDOS", "ENPH", "ROKU", "TXT",  "PKG", "MAA", "EPAM", "SNA",
    "LII", "AEG", "GDDY", "JBL", "FWONK",  "LW", "CNHI", "JHX",
    "MRO", "GEN",  "SHG", "ESS", "WPC", "ERIE", "SYF", "CSL", "SMCI",
     "SWK", "SQM", "CF", "MANH", "MAS",  "SSNC",  "TER",
    "TME", "GGG", "CAG", "DPZ", "LOGI", "POOL", "NDSN", "CFG",
    "AMCR", "IHG", "RPM",  "ELS", "MGM", "VST", "PODD", "ALB",
    "FWONA", "LNT", "GFI", "AZPN", "UAL", "AMH", "TKO", "NIO", "CG", "KIM",
    "NICE", "GLPI", "TAP",  "IP", "EDU",  "WSM", "BEN",
    "FNF", "HST",  "TWLO", "BIP",  "ACM", "NMR", "AGR", "INCY",
    "OC", "KEY", "H", "CELH", "ZBRA", "CPB", "OKTA", "MORN", "AES", "LKQ",
    "FMS", "REG",  "SJM", "VLYPO", "IPG", "EVRG", "JKHY", "EXAS",
    "CNA", "GL", "TRU", "NBIX", "BEP", "RDY", "LECO", "OVV", "RBA", "VTRS",
     "ESTC", "MOS", "TRMB", "TPL", "UHAL", "BCH", "UDR",  "AOS",
    "BURL", "UTHR",  "TEVA", "ARCC",  "SNN",  "DKS",
    "FIVE", "SAIA", "WES", "LBRDK", "LBRDA", "RVTY", "PAYC", "FLEX", "USFD", "OTEX",
    "NRG", "HTHT", "REXR", "PAA", "OZK", "PNR", "NI", "RGA", "NTNX", "TFX",
    "RNR",  "APA", "LAMR", "EQH", "CDAY", "FND", "CCK", "WRK",
    "DLB", "KMX", "PFGC", "Z",  "COTY", "CASY", "BAP", "CX", "TECH", "DOX", "CRL", "PSTG",
    "FFIV", "PAG", "PARA", "TFII", "PEAK","PR", "VLYPP", "EME", "CPT",
    "ZG", "CHRW", "EMN", "XPO", "BLD", "DBX", "WTRG", "ETSY", "DINO", "AFG",
    "BXP", "WMS",  "HII", "DVA", "SCI", "MKTX", "KEP",
    "QRVO", "CUBE", "SBS", "EWBC"
]


conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

symbol_count = len(symbols)
print("Total number of symbols:", symbol_count)


desired_row_count = 209

for symbol in symbols:
    df = yf.download(symbol, start='2019-01-01', end='2022-12-31', interval='1wk')
    if len(df) != desired_row_count:
        print(f"{symbol} does not have exactly 209 rows, it has {len(df)} rows. Skipping download.")
        continue
    df['Symbol'] = symbol
    df['Date'] = df.index
    df.set_index('Date', inplace=True)
    df.to_sql(symbol, conn, if_exists='replace', index=True)

conn.commit()
conn.close()



sector_mapping = {
    "FLT": "Consumer Discretionary", "CNP": "Utilities", "MKC": "Consumer Staples",
    "VIV": "Telecommunications", "PFG": "Finance", "DECK": "Consumer Discretionary",
    "ILMN": "Health Care", "WMG": "Consumer Discretionary", "TSN": "Consumer Staples",
    "WBA": "Consumer Staples", "BLDR": "Consumer Discretionary", "BMRN": "Health Care",
    "CHKP": "Technology", "EXPD": "Consumer Discretionary", "PHG": "Health Care",
    "CLX": "Consumer Discretionary", "AKAM": "Consumer Discretionary", "ZTO": "Industrials",
    "FITBI": "Finance", "AXON": "Industrials", "SIRI": "Consumer Discretionary",
    "TYL": "Technology", "TVE": "Utilities", "AQNB": "Utilities", "EG": "Finance",
    "VRT": "Technology", "HRL": "Consumer Staples", "RYAN": "Finance", "RPRX": "Health Care",
    "FDS": "Technology", "AGNCN": "Real Estate", "ATO": "Utilities", "YUMC": "Consumer Discretionary",
    "NOK": "Technology", "EDR": "Consumer Discretionary", "HBAN": "Finance", "WAT": "Industrials",
    "AER": "Consumer Discretionary", "LPLA": "Finance", "CMS": "Utilities", "NTRS": "Finance",
    "BAH": "Consumer Discretionary", "RIVN": "Consumer Discretionary", "WLK": "Industrials",
    "HOLX": "Health Care", "COO": "Health Care", "FSLR": "Technology", "ALGN": "Health Care",
    "ASBA": "Finance", "FITBP": "Finance", "SUI": "Real Estate", "LUV": "Consumer Discretionary",
    "CINF": "Finance", "OMC": "Consumer Discretionary", "STX": "Technology", "J": "Industrials",
    "HUBB": "Technology", "DT": "Technology", "AGNCM": "Real Estate", "SWKS": "Technology",
    "RF": "Finance", "VFS": "Finance", "EQT": "Energy", "ENTG": "Technology", "L": "Finance",
    "AGNCO": "Real Estate", "MGA": "Consumer Discretionary", "FITBO": "Finance",
    "WSO": "Consumer Discretionary", "AVY": "Consumer Discretionary", "SLMBP": "Finance",
    "RS": "Industrials", "BSY": "Technology", "BG": "Consumer Staples", "IEX": "Industrials",
    "CE": "Industrials", "KB": "Finance", "DGX": "Health Care", "WDC": "Technology",
    "SREA": "Consumer Discretionary", "LDOS": "Technology", "SOJE": "Basic Materials",
    "ENPH": "Technology", "ROKU": "Telecommunications", "TXT": "Industrials", "AQNU": "Utilities",
    "PKG": "Consumer Discretionary", "MAA": "Real Estate", "EPAM": "Technology",
    "SNA": "Consumer Discretionary", "LII": "Industrials", "AEG": "Finance", "GDDY": "Technology",
    "JBL": "Technology", "FWONK": "Consumer Discretionary", "AGNCP": "Real Estate",
    "LW": "Consumer Staples", "CNHI": "Industrials", "AGNCL": "Real Estate", "JHX": "Industrials",
    "MRO": "Energy", "GEN": "Technology", "FOXA": "Industrials", "SHG": "Finance",
    "ESS": "Real Estate", "WPC": "Real Estate", "ERIE": "Finance", "SYF": "Finance",
    "CSL": "Industrials", "SMCI": "Technology", "AVTR": "Industrials", "SWK": "Consumer Discretionary",
    "SQM": "Industrials", "CF": "Industrials", "MANH": "Technology", "MAS": "Consumer Discretionary",
    "PATH": "Technology", "SSNC": "Technology", "BKDT": "Health Care", "TER": "Industrials",
    "TME": "Consumer Discretionary", "PARAP": "Industrials", "BAM": "Real Estate",
    "GGG": "Industrials", "CAG": "Consumer Staples", "DPZ": "Consumer Discretionary",
    "LOGI": "Technology", "POOL": "Consumer Discretionary", "NDSN": "Industrials",
    "CFG": "Finance", "AMCR": "Consumer Discretionary", "IHG": "Consumer Discretionary",
    "RPM": "Consumer Discretionary", "XP": "Finance", "ELS": "Real Estate",
    "MGM": "Consumer Discretionary", "VST": "Utilities", "PODD": "Health Care",
    "ALB": "Industrials", "FOX": "Industrials", "FWONA": "Industrials", "LNT": "Utilities",
    "GFI": "Basic Materials", "AZPN": "Technology", "UAL": "Consumer Discretionary",
    "AMH": "Real Estate", "TKO": "Unknown", "NIO": "Consumer Discretionary", "CG": "Finance",
    "KIM": "Real Estate", "CRBG": "Finance", "NICE": "Technology", "GLPI": "Real Estate",
    "TAP": "Consumer Staples", "QRTEP": "Consumer Discretionary", "IP": "Basic Materials",
    "EDU": "Real Estate", "ACI": "Consumer Staples",    "WSM": "Consumer Discretionary", "BEN": "Finance", "FNF": "Finance",
    "HST": "Real Estate", "APP": "Technology", "TWLO": "Technology",
    "BIP": "Utilities", "PARAA": "Industrials", "ACM": "Consumer Discretionary",
    "NMR": "Finance", "AGR": "Utilities", "INCY": "Health Care",
    "OC": "Industrials", "KEY": "Finance", "H": "Consumer Discretionary",
    "CELH": "Consumer Staples", "ZBRA": "Industrials", "CPB": "Consumer Staples",
    "OKTA": "Technology", "MORN": "Finance", "AES": "Utilities",
    "LKQ": "Consumer Discretionary", "FMS": "Health Care", "REG": "Real Estate",
    "U": "Technology", "SJM": "Consumer Staples", "VLYPO": "Finance",
    "IPG": "Consumer Discretionary", "GRAB": "Consumer Discretionary", "EVRG": "Utilities",
    "JKHY": "Technology", "EXAS": "Health Care", "CNA": "Finance",
    "GL": "Finance", "TRU": "Finance", "NBIX": "Health Care",
    "BEP": "Utilities", "RDY": "Health Care", "LECO": "Industrials",
    "OVV": "Energy", "RBA": "Consumer Discretionary", "VTRS": "Health Care",
    "AFRM": "Consumer Discretionary", "ESTC": "Technology", "MOS": "Industrials",
    "TRMB": "Industrials", "TPL": "Energy", "UHAL": "Consumer Discretionary",
    "BCH": "Finance", "UDR": "Real Estate", "UGIC": "Utilities",
    "AOS": "Industrials", "BURL": "Consumer Discretionary", "UTHR": "Health Care",
    "BPYPP": "Finance", "TEVA": "Health Care", "ARCC": "Finance",
    "BPYPM": "Finance", "SNN": "Health Care", "TPG": "Finance",
    "DKS": "Consumer Discretionary", "BPYPO": "Finance", "FIVE": "Consumer Discretionary",
    "SAIA": "Industrials", "WES": "Utilities", "LBRDK": "Telecommunications",
    "LBRDA": "Telecommunications", "RVTY": "Industrials", "PAYC": "Technology",
    "FLEX": "Technology", "USFD": "Consumer Discretionary", "OTEX": "Technology",
    "NRG": "Utilities", "HTHT": "Consumer Discretionary", "REXR": "Real Estate",
    "PAA": "Energy", "OZK": "Finance", "PNR": "Industrials",
    "NI": "Utilities", "RGA": "Finance", "NTNX": "Technology",
    "TFX": "Health Care", "RNR": "Finance", "GFL": "Utilities",
    "APA": "Energy", "LAMR": "Real Estate", "EQH": "Finance",
    "CDAY": "Technology", "BPYPN": "Finance", "FND": "Consumer Discretionary",
    "CCK": "Industrials", "WRK": "Basic Materials", "DLB": "Technology",
    "KMX": "Consumer Discretionary", "PFGC": "Consumer Discretionary", "Z": "Consumer Discretionary",
    "LEGN": "Health Care", "COTY": "Consumer Discretionary", "CASY": "Consumer Discretionary",
    "BAP": "Finance", "CX": "Industrials", "TECH": "Health Care",
    "DOX": "Technology", "CRL": "Health Care", "PSTG": "Technology",
    "FFIV": "Technology", "PAG": "Consumer Discretionary", "PARA": "Industrials",
    "TFII": "Industrials", "PEAK": "Real Estate", "CHK": "Energy",
    "PR": "Energy", "VLYPP": "Finance", "EME": "Industrials",
    "CPT": "Real Estate", "ZG": "Consumer Discretionary", "CHRW": "Consumer Discretionary",
    "EMN": "Industrials", "XPO": "Consumer Discretionary", "BLD": "Consumer Discretionary",
    "DBX": "Technology", "WTRG": "Utilities", "ETSY": "Consumer Discretionary",
    "DINO": "Energy", "AFG": "Finance", "BXP": "Real Estate",
    "WMS": "Consumer Discretionary", "CHKEW": "Energy", "OZKAP": "Finance",
    "LCID": "Consumer Discretionary", "HII": "Industrials", "DVA": "Health Care",
    "SCI": "Consumer Discretionary", "MKTX": "Finance", "KEP": "Industrials",
    "QRVO": "Technology", "CUBE": "Real Estate", "SBS": "Utilities",
    "EWBC": "Finance"

}