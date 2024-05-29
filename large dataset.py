import yfinance as yf
import sqlite3

symbols = [
    'HDB', 'SCHW', 'ELV', 'BLK',
    'CAH', 'GS', 'RIO', 'LMT', 'SYK', 'BKNG', 'BUD', 'PLD', 'BCS', 'ISRG', 'SONY',
    'SBUX', 'TD', 'MUFG', 'MDT', 'DE', 'BMY', 'BP', 'TJX', 'AMT', 'MMC', 'MDLZ',
    'SHOP', 'UBS', 'PGR', 'PBR','ADP', 'EQNR', 'CB', 'LRCX', 'VRTX', 'ETN', 'ADI',
    'REGN', 'C',
     'IBN', 'ZTS', 'BX', 'SNPS', 'BSX', 'MELI', 'DEO', 'CME', 'FI',
    'SO', 'EQIX', 'CNI', 'CI', 'MO', 'ENB', 'GSK', 'ITW', 'INFY', 'RELX',
    'KLAC', 'SHW', 'CNQ', 'SLB', 'NOC', 'DUK', 'EOG', 'BALL', 'WDAY', 'VALE',
    'RACE', 'WM', 'STLA', 'MCO', 'GD', 'CP', 'BDX', 'RTO', 'SAN', 'HCA', 'TRI',
    'ANET', 'FDX', 'KKR', 'NTES', 'SMFG', 'CSX', 'ICE', 'AON', 'CL', 'BTI',
    'ITUB', 'PYPL', 'HUM', 'TGT', 'MCK', 'CMG', 'BMO', 'MAR', 'APD', 'AMX',
    'EPD', 'ORLY', 'E', 'ROP', 'MPC', 'PSX', 'MMM', 'CTAS', 'PH', 'BBVA',
    'LULU', 'BN', 'SCCO', 'HMC', 'PNC', 'APH', 'ECL', 'CHTR', 'MSI', 'BNS', 'NXPI',
    'TDG', 'AJG', 'PXD', 'ING', 'FCX', 'TT', 'APO', 'CCI', 'RSG', 'NSC',
    'OXY', 'EMR', 'DELL', 'TEAM', 'PCAR', 'PCG', 'WPP', 'AFL', 'WELL', 'MET',
     'EL', 'PSA','AZO', 'ADSK', 'CPRT', 'BSBR', 'AIG', 'DXCM', 'MCHP', 'ABEV',
    'KDP', 'ROST',
    'GM', 'CRH', 'SRE', 'PAYX', 'WMB', 'KHC', 'COF', 'MRVL', 'DHI',
    'STZ', 'TAK', 'ET', 'IDXX', 'ODFL', 'HLT', 'STM', 'VLO', 'SPG', 'HES',
    'F', 'MFG', 'DLR', 'TRV', 'EW', 'AEP', 'SU', 'MSCI', 'JD', 'KMB', 'COR',
    'NUE', 'LNG', 'OKE', 'FTNT', 'TEL', 'CNC', 'SQ', 'O',
    'BIDU', 'GWW', 'NEM', 'ADM', 'CM', 'TRP', 'IQV', 'KMI', 'D',
  'SPOT', 'HSY', 'EXC', 'LHX', 'GIS', 'A',
    'BK', 'JCI', 'EA', 'SYY', 'BCE', 'WDS',  'MPLX', 'ALL', 'WCN',
     'MFC', 'AME', 'AMP', 'FERG', 'BBD', 'PRU', 'FIS', 'CTSH',
     'YUM', 'FAST', 'VRSK', 'CSGP', 'LVS', 'IT', 'XEL', 'ARES', 'PPG',
     'TTD', 'IMO', 'BKR','HAL', 'CMI', 'URI', 'NDAQ', 'KR', 'ORAN', 'ROK', 'CVE', 'ED',
    'VICI', 'BBDO', 'PEG', 'ON', 'MDB', 'GPN', 'GOLD',
    'ACGL', 'DD', 'LYB', 'SLF', 'CHT', 'MRNA',  'PUK',
    'CQP', 'RCL', 'DG', 'ZS', 'IR', 'EXR', 'VEEV', 'CCEP', 'HPQ', 'MLM',
    'CDW', 'VMC', 'DVN', 'FICO', 'DLTR', 'EFX',  'PWR', 'FMX', 'TU', 'SBAC',
    'PKX', 'FANG', 'TTWO', 'MPWR', 'WBD', 'WEC', 'NTR', 'WIT', 'AEM',
    'VOD', 'ELP', 'EC', 'EIX', 'AWK', 'SPLK', 'XYL', 'ARGX', 'DB', 'WST',
    'HUBS', 'WTW', 'AVB', 'TEF', 'DFS', 'CBRE', 'TLK', 'KEYS', 'NWG', 'GLW', 'GIB',
    'ANSS', 'ZBH', 'DAL', 'HEI', 'SNAP', 'FTV',  'GRMN', 'HIG', 'RMD', 'RCI', 'MTD',
    'ULTA', 'CHD', 'IX', 'APTV', 'BR', 'WY', 'QSR', 'STT', 'TROW', 'TSCO',
    'VRSN', 'EQR', 'ICLR', 'DTE', 'RJF', 'MTB', 'WPM', 'CCL', 'EBAY', 'HWM', 'SE', 'MOH',
    'ALNY', 'WAB', 'TCOM','FE', 'ETR', 'FCNCA', 'BRO', 'ES',  'ARE', 'FNV', 'HPE', 'FITB', 'AEE',
    'INVH', 'CBOE', 'MT', 'NVR', 'TS', 'ROL', 'CCJ', 'DOV', 'FTS', 'STE', 'TRGP',
    'JBHT', 'UMC',  'EBR', 'IRM', 'BGNE', 'DRI', 'IFF', 'EXPE', 'PPL',
    'PTC', 'CTRA', 'TECK', 'TDY', 'VTR', 'WRB', 'STLD', 'GPC', 'ASX', 'LYV',
    'DUKB', 'NTAP',  'MKL', 'PBA', 'LH', 'KOF', 'K', 'ERIC', 'BAX', 'FLT',
    'CNP', 'MKC', 'VIV', 'PFG', 'DECK', 'ILMN', 'TSN', 'WBA', 'BLDR', 'BMRN',
    'CHKP', 'EXPD', 'PHG', 'CLX', 'AKAM', 'ZTO', 'FITBI', 'AXON', 'SIRI', 'TYL', 'TVE',
     'EG', 'VRT', 'HRL',  'FDS', 'AGNCN', 'ATO', 'YUMC', 'NOK',
     'HBAN', 'WAT', 'AER', 'LPLA', 'CMS', 'NTRS', 'BAH', 'WLK', 'HOLX',
    'COO', 'FSLR', 'ALGN',  'SUI', 'LUV', 'CINF', 'OMC', 'STX', 'J',
    'HUBB',   'SWKS', 'RF',  'EQT', 'ENTG', 'L',  'MGA',
     'WSO', 'AVY', 'SLMBP', 'RS',  'BG', 'IEX', 'CE', 'KB', 'DGX', 'WDC',
     'LDOS',  'ENPH', 'ROKU', 'TXT',  'PKG', 'MAA', 'EPAM', 'SNA',
    'LII', 'AEG', 'GDDY', 'JBL', 'FWONK',  'LW', 'CNHI',  'JHX', 'MRO',
    'GEN', 'SHG', 'ESS', 'WPC', 'ERIE', 'SYF', 'CSL', 'SMCI',  'SWK',
    'SQM', 'CF', 'MANH' , 'ELS', 'SSNC',  'TER', 'TME', 'GGG',
    'CAG', 'DPZ', 'LOGI', 'POOL', 'NDSN', 'CFG', 'AMCR', 'IHG','RPM'
]



# Create SQLite connection and cursor
conn = sqlite3.connect('large_stock_dataset.db')
cursor = conn.cursor()

symbol_count = len(symbols)
print("Total number of symbols:", symbol_count)


conn = sqlite3.connect('large_stock_dataset.db')

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
    "HDB": "Finance", "SCHW": "Finance", "ELV": "Health Care", "BLK": "Finance",
    "CAH": "Health Care", "GS": "Finance", "RIO": "Basic Materials", "LMT": "Industrials",
    "SYK": "Health Care", "BKNG": "Consumer Discretionary", "BUD": "Consumer Staples",
    "PLD": "Real Estate", "BCS": "Finance", "ISRG": "Health Care", "SONY": "Consumer Staples",
    "SBUX": "Consumer Discretionary", "TD": "Finance", "MUFG": "Finance", "MDT": "Health Care",
    "DE": "Industrials", "BMY": "Health Care", "BP": "Energy", "TJX": "Consumer Discretionary",
    "AMT": "Real Estate", "MMC": "Finance", "MDLZ": "Consumer Staples", "SHOP": "Technology",
    "UBS": "Finance", "PGR": "Finance", "PBR": "Energy", "ADP": "Technology", "EQNR": "Energy",
    "CB": "Finance", "PANW": "Technology", "LRCX": "Technology", "VRTX": "Technology",
    "ETN": "Technology", "ADI": "Technology", "REGN": "Health Care", "C": "Finance",
    "ABNB": "Consumer Discretionary", "IBN": "Finance", "ZTS": "Health Care", "BX": "Finance",
    "SNPS": "Technology", "BSX": "Health Care", "MELI": "Consumer Discretionary",
    "DEO": "Consumer Staples", "CME": "Finance", "FI": "Consumer Discretionary", "SO": "Utilities",
    "EQIX": "Real Estate", "CNI": "Industrials", "CI": "Health Care", "MO": "Consumer Staples",
    "ENB": "Energy", "GSK": "Health Care", "ITW": "Industrials", "INFY": "Technology",
    "RELX": "Consumer Discretionary", "KLAC": "Technology", "SHW": "Consumer Discretionary",
    "CNQ": "Energy", "SLB": "Energy", "NOC": "Industrials", "DUK": "Utilities",
    "EOG": "Energy", "BALL": "Industrials", "WDAY": "Technology", "VALE": "Basic Materials",
    "RACE": "Consumer Discretionary", "WM": "Utilities", "STLA": "Consumer Discretionary",
    "MCO": "Finance", "GD": "Industrials", "CP": "Industrials", "BDX": "Health Care",
    "RTO": "Consumer Discretionary", "SAN": "Finance", "HCA": "Health Care", "TRI": "Consumer Discretionary",
    "ANET": "Technology", "FDX": "Consumer Discretionary", "KKR": "Finance", "NTES": "Technology",
    "SMFG": "Finance", "CSX": "Industrials", "ICE": "Finance", "AON": "Finance",
    "CL": "Consumer Discretionary", "BTI": "Consumer Staples", "ITUB": "Finance",
    "PYPL": "Consumer Discretionary", "HUM": "Health Care", "TGT": "Consumer Discretionary",
    "MCK": "Health Care", "SNOW": "Technology", "CMG": "Consumer Discretionary", "BMO": "Finance",
    "MAR": "Consumer Discretionary", "APD": "Industrials", "AMX": "Telecommunications",
    "EPD": "Utilities", "ORLY": "Consumer Discretionary", "E": "Energy", "CRWD": "Technology",
    "ROP": "Industrials", "MPC": "Energy", "PSX": "Energy", "MMM": "Health Care",
    "CTAS": "Consumer Discretionary", "PH": "Industrials", "BBVA": "Finance",
    "LULU": "Consumer Discretionary", "BN": "Finance", "SCCO": "Basic Materials",
    "HMC": "Consumer Discretionary", "PNC": "Finance", "APH": "Technology",
    "ECL": "Consumer Discretionary", "CHTR": "Telecommunications", "MSI": "Technology",
    "BNS": "Finance", "NXPI": "Technology", "TDG": "Industrials", "AJG": "Finance",
    "PFH": "Finance", "PXD": "Energy", "ING": "Finance", "FCX": "Basic Materials",
    "TT": "Industrials", "APO": "Finance", "CCI": "Real Estate", "RSG": "Utilities",
    "NSC": "Industrials", "OXY": "Energy", "EMR": "Technology", "DELL": "Technology",
    "TEAM": "Technology", "PCAR": "Consumer Discretionary", "PCG": "Utilities",
    "WPP": "Consumer Discretionary", "AFL": "Finance", "WELL": "Real Estate", "MET": "Finance",
    "AESC": "Utilities", "EL": "Consumer Discretionary", "PSA": "Real Estate",
    "AZO": "Consumer Discretionary", "ADSK": "Technology", "CPRT": "Consumer Discretionary",
    "BSBR": "Finance", "AIG": "Finance", "DXCM": "Health Care", "MCHP": "Technology",
    "ABEV": "Consumer Staples", "KDP": "Consumer Staples", "ROST": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "CRH": "Industrials", "SRE": "Utilities",
    "PAYX": "Consumer Discretionary", "WMB": "Utilities", "CARR": "Industrials",
    "KHC": "Consumer Staples", "COF": "Finance", "MRVL": "Technology", "DHI": "Consumer Discretionary",
    "STZ": "Consumer Staples", "TAK": "Health Care", "ET": "Utilities", "IDXX": "Health Care",
    "ODFL": "Industrials", "HLT": "Consumer Discretionary", "STM": "Technology",
    "VLO": "Energy", "SPG": "Real Estate", "HES": "Energy", "F": "Consumer Discretionary",
    "MFG": "Finance", "DLR": "Real Estate", "TRV": "Finance", "EW": "Health Care",
    "AEP": "Utilities", "SU": "Energy", "MSCI": "Consumer Discretionary", "JD": "Consumer Discretionary",
    "KMB": "Consumer Staples", "COR": "Real Estate", "NUE": "Industrials", "SGEN": "Health Care",
    "LNG": "Utilities", "OKE": "Utilities", "FTNT": "Technology", "TEL": "Technology",
    "CNC": "Health Care", "SQ": "Technology", "PLTR": "Technology", "O": "Real Estate",
    "BIDU": "Technology", "GWW": "Industrials", "NEM": "Basic Materials", "ADM": "Consumer Staples",
    "CM": "Finance", "TRP": "Utilities", "IQV": "Health Care", "KMI": "Utilities",
    "DDOG": "Technology", "D": "Utilities", "KVUE": "Consumer Discretionary", "SPOT": "Consumer Discretionary",
    "HSY": "Consumer Staples", "EXC": "Utilities", "DASH": "Consumer Discretionary", "HLN": "Consumer Discretionary",
    "CEG": "Utilities", "LHX": "Industrials", "GIS": "Consumer Staples", "A": "Industrials",
    "BK": "Finance", "JCI": "Industrials", "EA": "Technology", "SYY": "Consumer Discretionary",
    "BCE": "Telecommunications", "WDS": "Energy", "LI": "Consumer Discretionary", "MPLX": "Energy",
    "ALL": "Finance", "WCN": "Utilities", "ALC": "Health Care", "MFC": "Consumer Discretionary",
    "AME": "Industrials", "DOW": "Industrials", "AMP": "Finance", "FERG": "Miscellaneous",
    "BBD": "Finance", "PRU": "Finance", "FIS": "Consumer Discretionary", "CTSH": "Technology",
    "OTIS": "Technology", "YUM": "Consumer Discretionary", "FAST": "Consumer Discretionary",
    "VRSK": "Technology", "CSGP": "Consumer Discretionary", "LVS": "Consumer Discretionary",
    "IT": "Consumer Discretionary", "XEL": "Utilities", "ARES": "Finance", "PPG": "Consumer Discretionary",
    "COIN": "Finance", "TTD": "Technology", "IMO": "Energy", "BKR": "Industrials",
    "HAL": "Energy", "CMI": "Industrials", "URI": "Industrials", "NDAQ": "Finance",
    "KR": "Consumer Staples", "ORAN": "Telecommunications", "ROK": "Industrials", "CVE": "Energy",
    "ED": "Utilities", "SATX": "Technology", "DKNG": "Consumer Discretionary", "VICI": "Real Estate",
    "BBDO": "Finance", "PEG": "Utilities", "ON": "Technology", "MDB": "Technology",
    "CTVA": "Industrials", "GEHC": "Technology", "GPN": "Consumer Discretionary", "GOLD": "Basic Materials",
    "ACGL": "Finance", "DD": "Industrials", "LYB": "Industrials", "SYM": "Industrials", "HBANM": "Finance", "SLF": "Finance", "CHT": "Telecommunications",
    "MRNA": "Health Care", "NU": "Finance", "PUK": "Finance", "CQP": "Utilities",
    "RCL": "Consumer Discretionary", "DG": "Consumer Discretionary", "ZS": "Technology",
    "IR": "Industrials", "EXR": "Real Estate", "VEEV": "Technology", "CCEP": "Consumer Staples",
    "HPQ": "Technology", "MLM": "Industrials", "GFS": "Technology", "CDW": "Consumer Discretionary",
    "VMC": "Industrials", "DVN": "Energy", "FICO": "Consumer Discretionary", "DLTR": "Consumer Discretionary",
    "EFX": "Finance", "CPNG": "Consumer Discretionary", "PWR": "Industrials", "FMX": "Consumer Staples",
    "TU": "Telecommunications", "SBAC": "Real Estate", "PKX": "Industrials", "FANG": "Energy",
    "TTWO": "Technology", "MPWR": "Technology", "WBD": "Telecommunications", "WEC": "Utilities",
    "NTR": "Industrials", "WIT": "Technology", "AEM": "Basic Materials", "HBANP": "Finance",
    "NET": "Technology", "VOD": "Telecommunications", "ELP": "Utilities", "EC": "Energy",
    "EIX": "Utilities", "AWK": "Utilities", "SPLK": "Technology", "XYL": "Industrials",
    "ARGX": "Health Care", "BNH": "Real Estate", "DB": "Finance", "WST": "Health Care",
    "RBLX": "Technology", "HUBS": "Technology", "WTW": "Finance", "AVB": "Real Estate",
    "TEF": "Telecommunications", "DFS": "Finance", "CBRE": "Finance", "TLK": "Telecommunications",
    "KEYS": "Industrials", "BNJ": "Real Estate", "NWG": "Finance", "GLW": "Technology",
    "GIB": "Consumer Discretionary", "ANSS": "Technology", "ZBH": "Health Care", "DAL": "Consumer Discretionary",
    "HEI": "Industrials", "SNAP": "Technology", "FTV": "Industrials", "BNTX": "Health Care",
    "GRMN": "Industrials", "HIG": "Finance", "RMD": "Health Care", "RCI": "Telecommunications",
    "MTD": "Industrials", "ULTA": "Consumer Discretionary", "CHD": "Consumer Discretionary",
    "IX": "Finance", "PINS": "Technology", "APTV": "Consumer Discretionary", "BR": "Consumer Discretionary",
    "WY": "Real Estate", "QSR": "Consumer Discretionary", "STT": "Finance", "TROW": "Finance",
    "TSCO": "Consumer Discretionary", "TW": "Finance", "VRSN": "Technology", "EQR": "Real Estate",
    "ICLR": "Health Care", "DTE": "Utilities", "RJF": "Finance", "MTB": "Finance",
    "WPM": "Basic Materials", "CCL": "Consumer Discretionary", "EBAY": "Consumer Discretionary",
    "HWM": "Industrials", "SE": "Consumer Discretionary", "MOH": "Health Care", "ALNY": "Health Care",
    "WAB": "Industrials", "TCOM": "Consumer Discretionary", "FE": "Utilities", "ETR": "Utilities",
    "FCNCA": "Finance", "BRO": "Finance", "ES": "Utilities", "ZM": "Technology",
    "ARE": "Real Estate", "FNV": "Basic Materials", "HPE": "Telecommunications", "FITB": "Finance",
    "AEE": "Utilities", "INVH": "Finance", "CBOE": "Finance", "MT": "Industrials",
    "NVR": "Consumer Discretionary", "TS": "Industrials", "ROL": "Finance", "CCJ": "Basic Materials",
    "DOV": "Industrials", "FTS": "Utilities", "STE": "Health Care", "TRGP": "Utilities",
    "JBHT": "Industrials", "UMC": "Technology", "RKT": "Finance", "BEKE": "Finance",
    "EBR": "Utilities", "IRM": "Real Estate", "BGNE": "Health Care", "DRI": "Consumer Discretionary",
    "IFF": "Industrials", "EXPE": "Consumer Discretionary", "PPL": "Utilities", "PTC": "Technology",
    "CTRA": "Energy", "TECK": "Basic Materials", "TDY": "Industrials", "VTR": "Real Estate",
    "WRB": "Finance", "STLD": "Industrials", "GPC": "Consumer Discretionary", "ASX": "Technology",
    "LYV": "Consumer Discretionary", "OWL": "Finance", "DUKB": "Utilities", "NTAP": "Technology",
    "VLTO": "Industrials", "IOT": "Technology", "MKL": "Finance", "PBA": "Energy",
    "LH": "Health Care", "KOF": "Consumer Staples", "K": "Consumer Staples", "ERIC": "Technology",
    "BAX": "Health Care", "FLT": "Consumer Discretionary", "CNP": "Utilities", "MKC": "Consumer Staples",
    "VIV": "Telecommunications", "PFG": "Finance", "DECK": "Consumer Discretionary", "ILMN": "Health Care",
    "WMG": "Consumer Discretionary", "TSN": "Consumer Staples", "WBA": "Consumer Staples", "BLDR": "Consumer Discretionary",
    "BMRN": "Health Care", "CHKP": "Technology", "EXPD": "Consumer Discretionary", "PHG": "Health Care",
    "CLX": "Consumer Discretionary", "AKAM": "Consumer Discretionary", "ZTO": "Industrials", "FITBI": "Finance",
    "AXON": "Industrials", "SIRI": "Consumer Discretionary", "TYL": "Technology", "TVE": "Utilities",
    "AQNB": "Utilities", "EG": "Finance", "VRT": "Technology", "HRL": "Consumer Staples",
    "RYAN": "Finance", "RPRX": "Health Care", "FDS": "Technology", "AGNCN": "Real Estate",
    "ATO": "Utilities", "YUMC": "Consumer Discretionary", "NOK": "Technology", "EDR": "Consumer Discretionary",
    "HBAN": "Finance", "WAT": "Industrials", "AER": "Consumer Discretionary", "LPLA": "Finance",
    "CMS": "Utilities", "NTRS": "Finance", "BAH": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "WLK": "Industrials", "HOLX": "Health Care", "COO": "Health Care", "FSLR": "Technology",
    "ALGN": "Health Care", "ASBA": "Finance", "FITBP": "Finance", "SUI": "Real Estate",
    "LUV": "Consumer Discretionary", "CINF": "Finance", "OMC": "Consumer Discretionary", "STX": "Technology",
    "J": "Industrials", "HUBB": "Technology", "DT": "Technology", "AGNCM": "Real Estate",
    "SWKS": "Technology", "RF": "Finance", "VFS": "Finance", "EQT": "Energy",
    "ENTG": "Technology", "L": "Finance", "AGNCO": "Real Estate", "MGA": "Consumer Discretionary",
    "FITBO": "Finance", "WSO": "Consumer Discretionary", "AVY": "Consumer Discretionary", "SLMBP": "Finance",
    "RS": "Industrials", "BSY": "Technology", "BG": "Consumer Staples", "IEX": "Industrials",
    "CE": "Industrials", "KB": "Finance", "DGX": "Health Care", "WDC": "Technology",
    "SREA": "Consumer Discretionary", "LDOS": "Technology", "SOJE": "Basic Materials", "ENPH": "Technology",
    "ROKU": "Telecommunications", "TXT": "Industrials", "AQNU": "Utilities", "PKG": "Consumer Discretionary",
    "MAA": "Real Estate", "EPAM": "Technology", "SNA": "Consumer Discretionary", "LII": "Industrials",
    "AEG": "Finance", "GDDY": "Technology", "JBL": "Technology", "FWONK": "Consumer Discretionary",
    "AGNCP": "Real Estate", "LW": "Consumer Staples", "CNHI": "Industrials", "AGNCL": "Real Estate",
    "JHX": "Industrials", "MRO": "Energy", "GEN": "Technology", "FOXA": "Industrials",
    "SHG": "Finance", "ESS": "Real Estate", "WPC": "Real Estate", "ERIE": "Finance",
    "SYF": "Finance", "CSL": "Industrials", "SMCI": "Technology", "AVTR": "Industrials",
    "SWK": "Consumer Discretionary", "SQM": "Industrials", "CF": "Industrials", "MANH": "Technology",
    "MAS": "Consumer Discretionary", "PATH": "Technology", "SSNC": "Technology", "BKDT": "Health Care",
    "TER": "Industrials", "TME": "Consumer Discretionary", "PARAP": "Industrials", "BAM": "Real Estate",
    "GGG": "Industrials", "CAG": "Consumer Staples", "DPZ": "Consumer Discretionary", "LOGI": "Technology",
    "POOL": "Consumer Discretionary", "NDSN": "Industrials", "CFG": "Finance", "AMCR": "Consumer Discretionary",
    "IHG": "Consumer Discretionary", "RPM": "Consumer Discretionary"}

