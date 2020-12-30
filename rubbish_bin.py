import pandas as pd
import numpy as np
from sklearn import preprocessing

def standardize_v1(df):
  dfRes = pd.DataFrame()
  scalersDict = {}
  cols = df.columns.values.tolist()
  for col in cols:
    dfCol = df.loc[:, [col]]
    scaler = preprocessing.StandardScaler()
    scaler.fit(dfCol)
    dfColT = scaler.transform(dfCol)
    scalersDict[col] = scaler
    dfRes[col] = dfColT.loc[:, col]
  res = {
    "df": dfRes,
    "scalersDict": scalersDict
  }
  return res