from db_utils import parse_cmd_arguments, save_to_pickle, load_from_pickle
import pyodbc
import pandas as pd
import pandas.io.sql as pds
import os


# SQL query for Optima db
query = '''WITH BnkZdarzenia AS (
	SELECT BZd_BZdID, BZd_Kierunek, BZd_PodmiotID, BZd_Rozliczono, BZd_PodmiotTyp, BZd_Termin, BZd_KwotaSys, BZd_DataDok, BZd_Numer, BZd_KwotaRozSys, BZd_DataRoz,
	Pod_Wojewodztwo, Pod_PodmiotTyp, Pod_Nazwa1, Pod_NIP, Pod_Regon
	FROM cdn.BnkZdarzenia
	INNER JOIN cdn.PodmiotyView
	ON (BZd_PodmiotID = Pod_PodId AND BZd_PodmiotTyp = Pod_PodmiotTyp)
),
BnkRozKwoty AS (
	SELECT BnkZdarzenia.*, BRK_PDokTyp, BRK_PDokID, BRK_LDokTyp, BRK_LDokID, BRK_Zwloka
	FROM BnkZdarzenia 
	LEFT OUTER JOIN cdn.BnkRozKwoty ON BRK_LDokID = BZd_BZdID 
	WHERE BRK_LDokTyp = 1
	UNION ALL
	SELECT BnkZdarzenia.*, BRK_PDokTyp, BRK_PDokID, BRK_LDokTyp, BRK_LDokID, BRK_Zwloka 
	FROM BnkZdarzenia 
	LEFT OUTER JOIN cdn.BnkRozKwoty ON BRK_PDokID = BZd_BZdID 
	WHERE BRK_PDokTyp = 1
),
BnkZapisy AS (
	SELECT BnkRozKwoty.*, BZp_DataDok, BZp_KwotaSys, BZp_KwotaRozSys 
	FROM BnkRozKwoty 
	LEFT OUTER JOIN cdn.BnkZapisy ON BRK_PDokID = BZp_BZpID 
	WHERE BRK_PDokTyp = 2
	UNION ALL
	SELECT BnkRozKwoty.*, BZp_DataDok, BZp_KwotaSys, BZp_KwotaRozSys  
	FROM BnkRozKwoty 
	LEFT OUTER JOIN cdn.BnkZapisy ON BRK_LDokID = BZp_BZpID 
	WHERE BRK_LDokTyp = 2
),
BnkDokNag AS (
SELECT * FROM cdn.BnkDokElem 
INNER JOIN cdn.BnkDokNag 
ON BDE_BDNId = BDN_BDNId 
WHERE bdn_typ=222 AND (BDE_DokTyp = 1 OR BDE_DokTyp = 12)
)

SELECT BZd_PodmiotID as PodmiotID,
BZd_PodmiotTyp as PodmiotTyp,
BZd_Termin as Termin, 
BZd_KwotaSys as KwotaSys, 
BZd_KwotaRozSys as KwotaRozSys,
BZd_DataRoz as DataRoz,
BDN_DataDok as DataPonag, 
BDN_TerminKst as TerminPonag, 
BDN_RazemKwotaSys1 as KwotaPonag,
BZp_DataDok as DataZap, 
BZp_KwotaSys as KwotaZap, 
BZp_KwotaRozSys as KwotaZapRoz,
BRK_Zwloka as Zwloka, 
BRK_PDokTyp as Dok1Typ, 
BRK_PDokID as Dok1ID, 
BRK_LDokTyp as Dok2Typ, 
BRK_LDokID as Dok2ID, 
Pod_Wojewodztwo as Wojewodztwo, 
Pod_Nazwa1 as Nazwa, 
Pod_NIP as NIP, 
Pod_Regon as Regon 
FROM BnkZapisy
LEFT OUTER JOIN BnkDokNag ON (BZd_BZdID = BDE_DokId AND BZd_PodmiotID = BDN_PodmiotID)
WHERE BZd_Kierunek = 1 AND BZd_Rozliczono <> 0
ORDER BY PodmiotID, PodmiotTyp, Dok1ID, DataZap, DataPonag'''

# SQL query for XL db
queryXL = '''WITH BnkFaktury AS (
SELECT TrP_GIDNumer, TrP_GIDTyp, TrP_GIDLp, TrP_Typ, TrP_KntNumer, TrP_KntTyp, TrP_Rozliczona, DATEADD(day,TrN_Data2,CONVERT(DATETIME,'1800-12-28',120)) as TrN_DataWys, DATEADD(day,TrP_Termin,CONVERT(DATETIME,'1800-12-28',120)) as TrP_Termin, TrP_KwotaSys, (TrP_KwotaSys-TrP_PozostajeSys) as Trp_KwotaRozSys
FROM cdn.TraNag
INNER JOIN cdn.TraPlat
ON (TrN_GIDTyp=TrP_GIDTyp AND TrN_GIDNumer=TrP_GIDNumer)
),
BnkZdarzenia AS (
	SELECT BnkFaktury.*,Knt_Wojewodztwo, Knt_GIDTyp, Knt_Nazwa1, Knt_Nip, Knt_Regon, Knt_Akronim, Knt_Dzialalnosc, Knt_Akwizytor 
	FROM BnkFaktury
	INNER JOIN cdn.KntKarty
	ON (Trp_KntNumer = Knt_GIDNumer AND Trp_KntTyp = Knt_GIDTyp)
),
BnkRozKwoty AS (
	SELECT BnkZdarzenia.*, R2_Dok1Typ, R2_Dok1Numer, R2_Dok2Typ, R2_Dok2Numer, DATEADD(day,R2_DataRozliczenia,CONVERT(DATETIME,'1800-12-28',120)) as DataRozliczenia
	FROM BnkZdarzenia
	LEFT OUTER JOIN cdn.Rozliczenia ON (TrP_GIDTyp=R2_Dok1Typ AND TrP_GIDNumer=R2_Dok1Numer AND TrP_GIDLp=R2_Dok1Lp)
	WHERE (R2_Dok1Typ<>784 AND R2_Dok1Typ<>0 AND R2_Dok1Typ<>435 AND R2_Dok1Typ<>1824 AND R2_Dok1Typ<>2034)
	UNION ALL
	SELECT BnkZdarzenia.*, R2_Dok1Typ, R2_Dok1Numer, R2_Dok2Typ, R2_Dok2Numer, DATEADD(day,R2_DataRozliczenia,CONVERT(DATETIME,'1800-12-28',120)) as DataRozliczenia
	FROM BnkZdarzenia
	LEFT OUTER JOIN cdn.Rozliczenia ON (TrP_GIDTyp=R2_Dok2Typ AND TrP_GIDNumer=R2_Dok2Numer AND TrP_GIDLp=R2_Dok2Lp)
	WHERE (R2_Dok2Typ<>784 AND R2_Dok2Typ<>0 AND R2_Dok2Typ<>435 AND R2_Dok2Typ<>1824 AND R2_Dok2Typ<>2034)
),
BnkZapisy AS (
	SELECT BnkRozKwoty.*, DATEADD(day,KAZ_DataZapisu,CONVERT(DATETIME,'1800-12-28',120)) as KAZ_DataZapisu, KAZ_KwotaSys, (KAZ_KwotaSys-KAZ_PozostajeSys) as KAZ_KwotaRozSys
	FROM BnkRozKwoty 
	LEFT OUTER JOIN cdn.Zapisy ON KAZ_GIDNumer=R2_Dok1Numer
	WHERE R2_Dok1Typ=784
	UNION ALL
	SELECT BnkRozKwoty.*, DATEADD(day,KAZ_DataZapisu,CONVERT(DATETIME,'1800-12-28',120)) as KAZ_DataZapisu, KAZ_KwotaSys, (KAZ_KwotaSys-KAZ_PozostajeSys) as KAZ_KwotaRozSys
	FROM BnkRozKwoty 
	LEFT OUTER JOIN cdn.Zapisy ON KAZ_GIDNumer=R2_Dok2Numer
	WHERE R2_Dok2Typ=784
),
BnkDokNag AS (
SELECT * FROM cdn.UpoElem
INNER JOIN cdn.UpoNag
ON (UPN_GIDTyp=UPE_GIDTyp AND UPN_GIDNumer=UPE_GIDNumer)
)

SELECT Trp_KntNumer as PodmiotID, 
Trp_KntTyp as PodmiotTyp,
TrN_DataWys as DataWys,
TrP_Termin as Termin,
DATEDIFF(d, TrN_DataWys, TrP_Termin) as TerminPlatnosci,
TrP_KwotaSys as KwotaSys, 
Trp_KwotaRozSys as KwotaRozSys, 
DataRozliczenia as DataRoz,
DATEADD(day,UPN_DataUp,CONVERT(DATETIME,'1800-12-28',120)) as DataPonag,
UPE_KwotaZal as KwotaPonag, 
KAZ_DataZapisu as DataZap, 
KAZ_KwotaSys as KwotaZap, 
KAZ_KwotaRozSys as KwotaZapRoz,
DATEDIFF(d, TrP_Termin, DataRozliczenia) as Zwloka,
R2_Dok1Typ as Dok1Typ, 
R2_Dok1Numer as Dok1ID, 
R2_Dok2Typ as Dok2Typ, 
R2_Dok2Numer as Dok2ID,
Knt_Wojewodztwo as Wojewodztwo, 
Knt_Nazwa1 as Nazwa, 
Knt_Akronim as Akronim,
Knt_Dzialalnosc as Dzialalnosc,
Knt_Akwizytor as Partner,
Knt_Nip as NIP, 
Knt_Regon as Regon 
FROM BnkZapisy
LEFT OUTER JOIN BnkDokNag ON (UPN_GIDTyp=TrP_GIDTyp AND UPN_GIDNumer=TrP_GIDNumer)
WHERE TrP_Typ = 2 AND TrP_Rozliczona <> 2
ORDER BY Trp_KntNumer, Trp_KntTyp, R2_Dok1Numer, KAZ_DataZapisu, UPN_DataUp'''

def db_connection(c,m='XL'):
    try:
        if m == 'XL':
            conn = pyodbc.connect('Driver={SQL Server};'  # Driver name
                                  'Server=FREJA;'  # Server name
                                  f'Database=cdnxl_blachotrapez_zoo;' # Database name
                                  'Trusted_Connection=yes;')
        elif m == 'Optima':
            conn = pyodbc.connect('Driver={SQL Server};'  # Driver name
                                  'Server=NBMWOSINSKI;'  # Server name
                                  f'Database=CDN_{c};' # Database name
                                  'Trusted_Connection=yes;')            
        return conn
    except pyodbc.DatabaseError as ex:
        sqlstate = ex.args[1]
        print(sqlstate) 
        print("Could not connect to database.")
        exit(1)


def invoices(u, m):
    args = parse_cmd_arguments()
    _frames = []
    
    if m=='XL':
        q=queryXL
        databases=['Blachotrapez']
    elif m=='Optima':
        q=query
        databases = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                     '13', '15', '16', '17', '18', '19', '21']
    #if args['update']:
    if u==True:
        user_id = 1
        for client in databases:
            conn = db_connection(c=client, m=m)
             
            # Data frame storing the SQL query
            df = pd.DataFrame(pds.read_sql(q, conn))
            if df.empty:
                continue

            df['db'] = client
            df['id'] = df.groupby(['PodmiotID', 'PodmiotTyp']).ngroup()
            df['id'] = df['id'] + user_id
            user_id = int(df['id'].iloc[-1])
            # Adding a data frame to the frames list
            _frames.append(df)
            save_to_pickle(df,'\\df_' + client, mode='XL')

        # df_cyclic = pd.concat(_frames, ignore_index=True)
    else:
        for df_pickled in os.listdir(os.getcwd() + f"\\dataframes\\" + m):
            df = load_from_pickle(df_pickled, mode=m)
            _frames.append(df)
    # Joining all tables into one large one
    df_cyclic = pd.concat(_frames, ignore_index=True)
    return df_cyclic



# =============================================================================
# df_all = invoices(u=False, m='XL')
# 
# df_termin = df_all[df_all['TerminPlatnosci']!=0]
# df_notermin = df_all[df_all['TerminPlatnosci']==0]
# 
# df_notermin = df_notermin.sample(int(len(df_notermin) * 0.1))
# 
# df_filtered = pd.concat([df_termin, df_notermin]).sort_index()
# 
# print(len(df_termin)/len(df_all))
# 
# print(df_all['TerminPlatnosci'].value_counts()[:10])
# print(round(df_all.TerminPlatnosci.value_counts(normalize=True),4)[:10]*100)
# 
# 
# print(len(df_notermin[df_notermin['Zwloka']>0])/len(df_notermin))
# print(len(df_all[df_all['Zwloka']>0]) / len(df_all))
# 
# df_ostatnie = df_all.groupby(['id'])['PodmiotID', 'PodmiotTyp', 'DataWys', 'Termin', 'TerminPlatnosci',
#        'KwotaSys', 'KwotaRozSys', 'DataRoz', 'DataPonag', 'KwotaPonag',
#        'DataZap', 'KwotaZap', 'KwotaZapRoz', 'Zwloka', 'Dok1Typ', 'Dok1ID',
#        'Dok2Typ', 'Dok2ID', 'Wojewodztwo', 'Nazwa', 'NIP', 'Regon', 'db',
#        'id'].last()
# =============================================================================







