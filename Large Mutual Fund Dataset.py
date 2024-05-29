import yfinance as yf
import pandas as pd
import sqlite3


symbols = [
    "ZGFIX", "FDVLX", "GNSRX", "FSLBX", "CRIMX", "NWKCX", "NWHZX", "GGEAX", "RRTRX", "DODGX",
    "FGIOX", "FERIX", "GWGVX", "PSCZX", "FEATX", "CWCFX", "GWGZX", "JSCRX", "PGOAX", "VSEQX",
    "TLVCX", "FSHCX", "PSCCX", "GSXIX", "CCVAX", "FSEAX",  "FEAAX", "PSCJX", "BRWIX",
    "GSCIX", "LCIAX", "GSXAX", "CSVIX", "JDMNX", "TLVAX", "JANEX", "DHLSX",
    "FERCX",  "FIJPX", "FIQCX", "FIQPX", "FGIZX", "TNBRX", "JGRTX", "DIAYX", "JAENX",
    "JDMAX", "SGIIX", "PWJQX", "FEGRX", "FAMRX", "TNBIX", "JMGRX", "PWJRX", "DIAMX", "HWCAX",
    "LGMAX", "TAMVX", "TMVIX", "CRMMX",  "FZAJX", "CSCCX", "FESGX", "FIFFX", "WSMNX",
    "GSXCX", "JDMRX", "MSFLX", "JGRCX", "VHCOX",  "FVIFX", "FVLZX", "RRTDX", "MSFBX",
    "TRSSX", "SGENX", "THOFX", "LGMCX", "LSWWX","LGMNX", "THOGX", "GCPNX", "PWJBX", "RPMAX",
    "PWJAX", "FNSDX", "FDEEX","FGRIX", "CCGSX", "FEYCX", "PSCHX", "PWJCX", "VHCAX", "MSGFX",


    "PARDX", "OTCFX", "HMXIX", "FTVFX", "OTIIX", "FCPCX", "ZGFAX", "FJPNX",
    "NWAMX",  "LCORX", "PWJDX", "CCGIX", "PASSX", "FEYTX", "THORX", "AFCSX",
    "FCPAX", "FAVFX", "LCRIX", "TRRDX", "THOVX", "FVLKX", "MSFAX", "MGISX", "PWJZX", "NEFOX",
    "AFCMX", "FEYAX", "HWCIX", "GWGIX", "SIBAX", "FGIKX", "FIVFX", "FCVFX", "THOAX", "NECOX",
    "TRMIX", "WSMDX", "WBSIX", "HWCCX", "FEYIX", "TRMCX", "SLVAX", "THOIX", "HMXAX", "FIATX",
    "AFCHX",  "HMXCX", "VPCCX", "WBSNX", "CSVZX", "FITGX",  "RRMVX", "FCPIX",
    "FIAGX", "FIDZX", "HWLCX", "FTFFX", "DFSGX", "SLVRX", "FIIIX", "SPINX", "THOCX", "CSERX",
    "HWLAX", "NEOYX", "FIGFX", "SVLCX", "CSRYX", "NOANX", "SSQSX", "HWLIX", "SIVIX",
    "FAFFX", "CAMWX", "FGTNX", "CAMOX", "AFCWX", "PORIX", "BGRSX", "FKGLX", "FIGCX", "AFVZX",
    "PORTX", "FOSFX", "AASMX", "SLVIX", "RPGIX", "AFCLX", "TRGAX", "FPJAX",  "BGLSX",


    "LSOFX", "TILCX", "FIQLX", "AFCNX", "VADFX", "VADAX", "POAGX", "FOSKX", "USPCX", "FJPIX",
    "QKBGX", "ABLOX", "FCFFX", "VADRX", "DFDSX", "USPFX", "VADDX", "PURRX", "GWEIX", "VADCX",
     "GWEZX", "BSGSX", "PVFAX", "AAUTX", "OLVAX", "OLVRX", "PGRQX", "PURZX", "TRRJX",
    "RCMFX", "MUNDX", "TLVIX", "BSGIX", "CBLRX", "GWETX", "VDIGX", "ECSTX", "SSSIX",
    "VPMAX", "CBDYX", "OLVCX",   "USPVX", "VGSAX",
     "VPMCX", "FGABX", "TSCSX", "FJPCX",
    "QCBGX", "JEQIX", "BLUEX", "SSSFX", "CBALX", "VRGEX", "CLREX", "VGISX", "CBLAX", "PURCX",
    "EXHAX", "VLSIX", "MNHIX", "FJPTX", "OLVTX", "CBDRX", "QABGX", "HLQVX",  "RRTPX",
    "PACLX", "QIBGX", "NRGSX", "NBGIX",  "NBGAX", "JLVMX", "COAGX", "VGSCX", "JLVZX",
    "ERSTX", "JLVRX", "NEAGX", "CBLCX", "EHSTX", "PRWCX", "TRAIX", "PARKX", "SEVSX", "EILVX",
    "ERLVX", "NBGEX", "PURAX", "DREGX", "SEVPX", "LKBAX", "NBGNX", "QLEIX", "VLSCX", "PUREX",
    "PCAFX", "PURGX", "NEAIX", "VSTCX", "CSRIX",  "SEVAX", "QLERX",
    "HHDFX", "FOBPX", "HHDVX", "FCGCX", "WCMSX", "FOBAX", "MNHRX", "GQGPX",
    "PHRAX", "VRREX","JANBX", "SEBLX", "SBACX",
    "FSCRX", "NEEGX"

]


conn = sqlite3.connect('large_mutual_data.db')

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

    "ZGFIX": "Ninety One",   "FDVLX": "Fidelity",  "GNSRX": "abrdn",  "FSLBX": "Fidelity",  "CRIMX": "CRM",  "NWKCX": "Nationwide",   "NWHZX": "Nationwide",
    "GGEAX": "Nationwide", "RRTRX": "T. Rowe Price",
    "DODGX": "Dodge & Cox",  "FGIOX": "Fidelity",   "FERIX": "Fidelity",   "GWGVX": "AMG",  "PSCZX": "PGIM",  "FEATX": "Fidelity",  "CWCFX": "Christopher Weil & Co",
    "GWGZX": "AMG", "JSCRX": "PGIM",  "PGOAX": "PGIM",  "VSEQX": "Vanguard",  "TLVCX": "Timothy",
    "FSHCX": "Fidelity",  "PSCCX": "PGIM",  "GSXIX": "abrdn",  "CCVAX": "Calvert",  "FSEAX": "Fidelity",  "FEAAX": "Fidelity",
    "PSCJX": "PGIM", "BRWIX": "AMG",  "GSCIX": "abrdn", "LCIAX": "SEI", "GSXAX": "abrdn", "CSVIX": "Calvert", "JDMNX": "Janus Henderson",   "TLVAX": "Timothy", "JANEX": "Janus Henderson",
    "DHLSX": "Diamond Hill",  "FERCX": "Fidelity",  "FIJPX": "Fidelity", "FIQPX": "Fidelity",  "FGIZX": "Fidelity",   "TNBRX": "1290 SmartBeta",
    "JGRTX": "Janus Henderson", "DIAYX": "Diamond Hill", "JAENX": "Janus Henderson", "JDMAX": "Janus Henderson",  "SGIIX": "First Eagle",
    "PWJQX": "PGIM", "FEGRX": "First Eagle",  "FAMRX": "Fidelity",  "TNBIX": "1290 SmartBeta","JMGRX": "Janus Henderson",
    "PWJRX": "PGIM", "DIAMX": "Diamond Hill", "HWCAX": "Hotchkis & Wiley",  "LGMAX": "Loomis Sayles",  "TAMVX": "T. Rowe Price", "TMVIX": "Timothy",
    "CRMMX": "CRM", "FZAJX": "Fidelity", "CSCCX": "Calvert", "FESGX": "First Eagle", "FIFFX": "Fidelity",  "WSMNX": "William Blair",   "GSXCX": "abrdn", "JDMRX": "Janus Henderson",
    "MSFLX": "Morgan Stanley", "JGRCX": "Janus Henderson","VHCOX": "Vanguard",  "FVIFX": "Fidelity",  "FVLZX": "Fidelity",  "RRTDX": "T. Rowe Price",  "MSFBX": "Morgan Stanley",   "TRSSX": "T. Rowe Price","SGENX": "First Eagle",
    "THOFX": "Thornburg",  "LGMCX": "Loomis Sayles", "LSWWX": "Loomis Sayles",  "LGMNX": "Loomis Sayles",
    "THOGX": "Thornburg",  "GCPNX": "Gateway",  "PWJBX": "PGIM",  "RPMAX": "Reinhart",  "PWJAX": "PGIM",   "FNSDX": "Fidelity", "FDEEX": "Fidelity",  "FGRIX": "Fidelity", "CCGSX": "Baird", "FEYCX": "Fidelity",
    "PSCHX": "PGIM", "PWJCX": "PGIM", "VHCAX": "Vanguard", "MSGFX": "Morgan Stanley",  "PARDX": "T. Rowe Price",
    "OTCFX": "T. Rowe Price",  "HMXIX": "AlphaCentric",  "FTVFX": "Fidelity",  "OTIIX": "T. Rowe Price", "FCPCX": "Fidelity", "ZGFAX": "Ninety One",
    "FJPNX": "Fidelity",   "NWAMX": "Nationwide",  "LCORX": "Leuthold Core",   "PWJDX": "PGIM",   "CCGIX": "Baird",  "PASSX": "T. Rowe Price",  "FEYTX": "Fidelity",  "THORX": "Thornburg",  "AFCSX": "American Century",  "FCPAX": "Fidelity",  "FAVFX": "Fidelity",   "LCRIX": "Leuthold","TRRDX": "T. Rowe Price",   "THOVX": "Thornburg",
    "FVLKX": "Fidelity", "MSFAX": "Morgan Stanley", "MGISX": "Morgan Stanley", "PWJZX": "PGIM", "NEFOX": "Natixis",
    "AFCMX": "American Century",    "FEYAX": "Fidelity",   "HWCIX": "Hotchkis & Wiley",   "GWGIX": "AMG",
    "SIBAX": "Sit Balanced",  "FGIKX": "Fidelity",  "FIVFX": "Fidelity",  "FCVFX": "Fidelity", "THOAX": "Thornburg",  "NECOX": "Natixis Oakmark",
    "TRMIX": "T. Rowe Price",  "WSMDX": "William Blair",  "WBSIX": "William Blair",  "HWCCX": "Hotchkis & Wiley",  "FEYIX": "Fidelity",  "TRMCX": "T. Rowe Price",  "SLVAX": "Columbia",
    "THOIX": "Thornburg","HMXAX": "AlphaCentric","FIATX": "Fidelity",   "AFCHX": "American Century",   "HMXCX": "AlphaCentric", "VPCCX": "Vanguard",    "WBSNX": "William Blair",
    "CSVZX": "Columbia",  "FITGX": "Fidelity",  "RRMVX": "T. Rowe Price",  "FCPIX": "Fidelity",    "FIAGX": "Fidelity",  "FIDZX": "Fidelity", "HWLCX": "Hotchkis & Wiley",
    "FTFFX": "Fidelity",  "DFSGX": "DF Dent",  "SLVRX": "Columbia",  "FIIIX": "Fidelity",   "SPINX": "SEI",   "THOCX": "Thornburg",
    "CSERX": "Columbia",    "HWLAX": "Hotchkis & Wiley",    "NEOYX": "Natixis",    "FIGFX": "Fidelity",    "SVLCX": "Columbia",
    "CSRYX": "Columbia",  "NOANX": "Natixis",  "SSQSX": "State Street",  "HWLIX": "Hotchkis & Wiley",  "SIVIX": "State Street",  "FAFFX": "Fidelity",
    "CAMWX": "Cambiar",  "FGTNX": "Fidelity",  "CAMOX": "Cambiar",  "AFCWX": "American Century",  "PORIX": "Trillium",  "BGRSX": "Boston Partners",
    "FKGLX": "Fidelity",  "FIGCX": "Fidelity",  "AFVZX": "Applied Finance",  "PORTX": "Trillium",  "FOSFX": "Fidelity",  "AASMX": "Thrivent",  "SLVIX": "Columbia",
    "RPGIX": "T. Rowe Price", "AFCLX": "American Century", "TRGAX": "T. Rowe Price", "FPJAX": "Fidelity",  "BGLSX": "Boston Partners",  "LSOFX": "LS Opportunity",
    "TILCX": "T. Rowe Price",  "FIQLX": "Fidelity",  "AFCNX": "American Century",  "VADFX": "Invesco",  "VADAX": "Invesco",  "POAGX": "PRIMECAP",  "FOSKX": "Fidelity",  "USPCX": "Union Street Partners", "FJPIX": "Fidelity",
    "QKBGX": "Federated",  "ABLOX": "Alger",  "FCFFX": "Fidelity",  "VADRX": "Invesco",  "DFDSX": "DF Dent",  "USPFX": "Union","VADDX": "Invesco",
    "PURRX": "PGIM",  "GWEIX": "AMG",  "VADCX": "Invesco",  "GWEZX": "AMG",  "BSGSX": "Baird",   "PVFAX": "Paradigm Value",
    "AAUTX": "Thrivent", "OLVAX": "JPMorgan", "OLVRX": "JPMorgan", "PGRQX": "PGIM", "PURZX": "PGIM", "TRRJX": "T. Rowe Price", "RCMFX": "Schwartz",
    "MUNDX": "Mundoval",  "TLVIX": "Thrivent",  "BSGIX": "Baird",  "CBLRX": "Columbia",  "GWETX": "AMG","VDIGX": "Vanguard",   "ECSTX": "Eaton Vance",  "SSSIX": "SouthernSun",
    "VPMAX": "Vanguard",  "CBDYX": "Columbia",  "OLVCX": "JPMorgan",  "USPVX": "Union Street Partners",  "VGSAX": "Virtus Duff & Phelps",  "VPMCX": "Vanguard",
    "FGABX": "Fidelity", "TSCSX": "Thrivent", "FJPCX": "Fidelity", "QCBGX": "Hermes", "JEQIX": "Johnson",
    "BLUEX": "AMG", "SSSFX": "SouthernSun", "CBALX": "Columbia", "VRGEX": "Virtus", "CLREX": "Columbia",    "VGISX": "Virtus Duff & Phelps","CBLAX": "Columbia",  "PURCX": "PGIM",  "EXHAX": "Manning & Napier",   "VLSIX": "Virtus",   "MNHIX": "Manning & Napier",
    "FJPTX": "Fidelity",  "OLVTX": "JPMorgan",  "CBDRX": "Columbia",  "QABGX": "Hermes",
    "HLQVX": "JPMorgan", "RRTPX": "T. Rowe Price",   "PACLX": "T. Rowe Price", "QIBGX": "Hermes",   "NRGSX": "Neuberger Berman",  "NBGIX": "Neuberger Berman", "NBGAX": "Neuberger Berman",
    "JLVMX": "JPMorgan", "COAGX": "Caldwell & Orkin", "VGSCX": "Virtus", "JLVZX": "JPMorgan", "ERSTX": "Eaton Vance",   "JLVRX": "JPMorgan",
    "NEAGX": "Needham","CBLCX": "Columbia","EHSTX": "Eaton Vance","PRWCX": "T. Rowe Price","TRAIX": "T. Rowe Price","PARKX": "T. Rowe Price",   "SEVSX": "Guggenheim",   "EILVX": "Eaton Vance",    "ERLVX": "Eaton Vance","NBGEX": "Neuberger Berman",
    "PURAX": "PGIM",  "DREGX": "Driehaus",  "SEVPX": "Guggenheim", "LKBAX": "LKCM",  "NBGNX": "Neuberger",
    "QLEIX": "AQR",  "VLSCX": "Virtus",  "PUREX": "PGIM",  "PCAFX": "Prospector",  "PURGX": "PGIM",  "NEAIX": "Needham", "VSTCX": "Vanguard", "AGVDX": "American Funds", "CSRIX": "Cohen & Steers",
    "CGVBX": "American Funds ", "SEVAX": "Guggenheim", "QLERX": "AQR", "CGVEX": "American Funds", "AGVFX": "American Funds","AGVEX": "American Funds", "CGVYX": "American Funds","RGLEX": "American Funds ",
    "HHDFX": "Hamlin", "FOBPX": "Tributary", "HHDVX": "Hamlin", "CSJCX": "Cohen & Steers", "FCGCX": "Fidelity", "WCMSX": "WCM","CSJIX": "Cohen & Steers",
    "CSRSX": "Cohen & Steers",  "CSJAX": "Cohen & Steers",  "CSJRX": "Cohen & Steers", "CSJZX": "Cohen & Steers", "FFGTX": "Fidelity", "MNHCX": "Manning & Napier","FOBAX": "Tributary","MNHRX": "Manning & Napier","GQGPX": "GQG Partners","PHRAX": "Virtus","VRREX": "Virtus","GQGIX": "GQG Partners",   "GQGRX": "GQG Partners",
    "FMIJX": "FMI",  "VLSAX": "Virtus",  "JDBAX": "Janus Henderson","GURCX": "Guggenheim", "FMIYX": "FMI",   "JABAX": "Janus Henderson",
    "JABNX": "Janus Henderson", "JBALX": "Janus Henderson", "SCVEX": "Hartford", "DIEMX": "Driehaus", "GURAX": "Guggenheim", "GURPX": "Guggenheim",  "VLSRX": "Virtus", "ICSIX": "Dynamic",  "ICSNX": "Dynamic",  "RYAVX": "Rydex",
    "EAFVX": "Eaton Vance",  "RYMVX": "Rydex", "VASGX": "Vanguard",  "GTSCX": "Glenmede",  "GURIX": "Guggenheim",  "EIFVX": "Eaton Vance",  "RAIWX": "Manning & Napier",  "JABCX": "Janus Henderson",
    "BBHLX": "BBH Partner Fund", "RYMMX": "Rydex", "RAIRX": "Manning & Napier",  "JDBRX": "Janus Henderson",  "UGTCX": "Victory Growth and Tax",
    "BTBFX": "Boston Trust Asset Management","JABRX": "Janus Henderson",   "UGTAX": "Victory Growth and Tax",  "UGTIX": "Victory Growth and Tax",  "JANBX": "Janus Henderson",  "SEBLX": "Touchstone",  "SBACX": "Touchstone",  "FSCRX": "Fidelity","NEEGX": "Needham"

}


symbol_count = len(company_mapping_filtered)
print("Total number of symbols:", symbol_count)


missing_symbols = [symbol for symbol in symbols if symbol not in company_mapping_filtered.keys()]

print(missing_symbols)
