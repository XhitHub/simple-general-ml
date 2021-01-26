import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.impute import *

# https://scikit-learn.org/stable/modules/preprocessing.html

possParams = {
  "scaler": [
    StandardScaler(),
    # scaler for when many outliners
    RobustScaler()
  ]
}

def run(df, scaler=StandardScaler()):
  print ("general preprocess")
  impute(df)
  scaler = preprocessNumeric(df, scaler)
  res = {
    "standardizeScaler": scaler
  }
  return res

def preprocess(df, preprocesser):
  df = preprocesser.fit_transform(df)
  return preprocesser

def preprocessNumeric(df, preprocesser):
  num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
  df[num_cols] = preprocesser.fit_transform(df[num_cols])
  return preprocesser

def preprocessCols(df, cols, preprocesser):
  df[cols] = preprocesser.fit_transform(df[cols])
  return preprocesser

def encodeCtgs(df, col, preprocesser=OneHotEncoder()):
  df2 = preprocesser.fit_transform(df[[col]])
  df2 = pd.DataFrame(df2.todense())
  ctgs = preprocesser.categories_[0]
  for dCol in df2.columns:
    df.loc[:,col + '_' + ctgs[dCol]] = df2[dCol]
  df.drop(col, 1, inplace=True)
  return preprocesser

def discretize(df, ctgCount):
  # print('discretize df nan check:')
  # print(df.isnull().values.any())
  est = KBinsDiscretizer(n_bins=ctgCount, strategy='uniform', encode='ordinal').fit(df)
  ndarrayTransformed = est.transform(df)
  # ndarrayTransformed2 = est.inverse_transform(ndarrayTransformed)
  resDf = pd.DataFrame(ndarrayTransformed, index=df.index, columns=df.columns)
  return resDf

def fillMissing(df):
  for col in df.columns:
    df[col].fillna(method='pad',inplace=True)

def impute(df, imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')):
  data = imputer.fit_transform(df)
  ndenum = np.ndenumerate(data)
  # print(ndenum)
  # print(len(ndenum))
  # print(df.shape)
  for (x,y), value in np.ndenumerate(data):
    # print(x)
    # print(y)
    # print(df.columns[y])
    # print(value)
    oVal = df.at[x, df.columns[y]]
    if (oVal == '' or pd.isnull(oVal)):
      df.at[x, df.columns[y]] = value

def finalImpute(df):
  df.fillna(value=0, inplace=True)


# def runWithListOfPreprocessers(df, numPreprocessers):
#   resList = []
#   for numP in numPreprocessers:
#     res = preprocessNumeric(df, numP)
#     resList.append(res)
#   return resList