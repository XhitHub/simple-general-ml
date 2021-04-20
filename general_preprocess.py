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

def scale(df):
  scaler = StandardScaler()
  scalerFitted = preprocessNumeric(df, scaler)
  return scalerFitted

def scaleNoFit(df, scaler):
  num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
  df[num_cols] = scaler.transform(df[num_cols])

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

def myDiscretize(df, ctgCount):
  processed = 0
  total = len(df.columns)
  for colName in df.columns:
    print('Processing myDiscretize on ' + str(colName) + ' (' + str(processed) +'/' + str(total) + ')')
    col = df[colName]
    valRange = col.max() - col.min()
    interval = valRange / ctgCount
    sectMax = [0] * ctgCount
    classes = [0] * ctgCount
    for i in range(0,ctgCount):
      sectMax[i] = col.min() + (i+1) * interval
      classMin = (str(col.min() + i * interval))[:8]
      classMax = (str(col.min() + (i+1) * interval))[:8]
      classAvg = (str(col.min() + (i+0.5) * interval))[:8]
      # classes[i] = "S" + str(col.min() + i * interval)
      classes[i] = str(i) + ", " + classAvg + ": " + classMin + " to " + classMax
    newCol = []
    # for val in col:
    for i in range(0, len(col)):
      for j in range(0,ctgCount):
        if sectMax[j] >= col[i]:
          col[i] = classes[j]
          break
    processed += 1
  return df

def percentageChangeToCtg(df, ctgArr):
  processed = 0
  total = len(df.columns)
  for colName in df.columns:
    print('Processing percentageChangeToCtg on ' + str(colName) + ' (' + str(processed) +'/' + str(total) + ')')
    col = df[colName]
    # for val in col:
    for i in range(0, len(col)):
      # *100 as is percentage
      val = col[i] * 100
      prefix = None
      resC = 0
      if val > 0:
        prefix = 'i'
        for c in ctgArr:
          if val >= c and c > resC:
            resC = c
      if val <= 0:
        prefix = 'd'
        for c in ctgArr:
          if -val >= c and c > resC:
            resC = c
      resVal = prefix + str(resC)
      col[i] = resVal
    processed += 1
  return df
    
def percentageChangeToOrdinalCtg(df, ctgArr):
  processed = 0
  total = len(df.columns)
  for colName in df.columns:
    print('Processing percentageChangeToCtg on ' + str(colName) + ' (' + str(processed) +'/' + str(total) + ')')
    col = df[colName]
    # for val in col:
    for i in range(0, len(col)):
      # *100 as is percentage
      val = col[i] * 100
      prefix = None
      resC = 0
      if val > 0:
        for c in ctgArr:
          if val >= c and c > resC:
            resC = c
      if val <= 0:
        for c in ctgArr:
          if -val >= c and c > resC:
            resC = -c
      resVal = resC
      col[i] = resVal
    processed += 1
  return df

def getValCtg(val, valCtgArr):
  # default is 0
  # print('getValCtg val: ' + val)
  valCtg = 0
  if (val > 0):
    for v in valCtgArr:
      if val > v:
        valCtg = v
      else:
        break
  if (val < 0):
    for v in valCtgArr:
      if val < v:
        valCtg = v
        break
  return valCtg

def getValCtgColName(colName, valCtg):
  postfix = '_c' + str(valCtg)
  return colName + postfix

def toValCtgDf(df, valCtgArr):
  dfRowsCount = len(df.index)
  dfColsCount = len(df.columns)
  # create resDf with same rows as df, cols being ctgs for each df col 
  # postfixes = ['c0']
  # for v in valCtgArr:
  #   postfixes.append('c' + str(v))
  if 0 not in valCtgArr:
    valCtgArr.append(0)
  valCtgArr.sort()
  # zero 2darr
  # arr2d = np.zeros((dfRowsCount, dfColsCount * len(postfixes)))
  arr2d = np.zeros((dfRowsCount, dfColsCount * len(valCtgArr)))
  # create new cols
  newCols = []
  for colName in df.columns:
    # for postfix in postfixes:
    #   newCols.append(colName + '_' + postfix)
    for valCtg in valCtgArr:
      newCols.append(getValCtgColName(colName, valCtg))
  # create new df
  resDf = pd.DataFrame(arr2d, columns=newCols)
  # fill 1 to resDf according to values
  for index, row in df.iterrows():
    print('toValCtgDf: ' + str(index) + '/' + str(len(df.index)))
    for colName in df.columns:
      # get ctg cell belongs to. need * 100 as is percentage
      valCtg = getValCtg(row[colName] * 100, valCtgArr)
      # fill 1 to corresponding col in resDf
      targetColName = getValCtgColName(colName, valCtg)
      resDf.at[index, targetColName] = 1
  return resDf

def discretize(df, ctgCount):
  # print('discretize df nan check:')
  # print(df.isnull().values.any())
  est = KBinsDiscretizer(n_bins=ctgCount, strategy='uniform', encode='ordinal').fit(df)
  ndarrayTransformed = est.transform(df)
  # resDf = pd.DataFrame(ndarrayTransformed, index=df.index, columns=df.columns)
  ndarrayTransformed2 = est.inverse_transform(ndarrayTransformed)
  resDf = pd.DataFrame(ndarrayTransformed2, index=df.index, columns=df.columns)
  return resDf

def fillMissing(df):
  for col in df.columns:
    df[col].fillna(method='pad',inplace=True)

def fillMissingFinal(df, value):
  nonNanCount = df.count()
  nanCount = df.isnull().sum().sum()
  print('fillMissingFinal still Nan count: ' + str(nanCount) + '/' + str(nonNanCount + nanCount))
  for col in df.columns:
    df[col].fillna(value=value,inplace=True)

def impute(df, imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')):
  imputerFitted = imputer.fit(df)
  data = imputerFitted.transform(df)
  ndenum = np.ndenumerate(data)
  for (x,y), value in np.ndenumerate(data):
    oVal = df.at[x, df.columns[y]]
    if (oVal == '' or pd.isnull(oVal)):
      df.at[x, df.columns[y]] = value
  return imputerFitted

def imputeNoFit(df, imputer):
  imputerFitted = imputer
  data = imputerFitted.transform(df)
  ndenum = np.ndenumerate(data)
  for (x,y), value in np.ndenumerate(data):
    oVal = df.at[x, df.columns[y]]
    if (oVal == '' or pd.isnull(oVal)):
      df.at[x, df.columns[y]] = value
  return imputerFitted

def finalImpute(df):
  df.fillna(value=0, inplace=True)


# def runWithListOfPreprocessers(df, numPreprocessers):
#   resList = []
#   for numP in numPreprocessers:
#     res = preprocessNumeric(df, numP)
#     resList.append(res)
#   return resList


# def impute(df, imputer):
#   if (imputer==None):
#     imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#     imputerFitted = imputer.fit(df)
#   else:
#     imputerFitted = imputer
#   data = imputerFitted.transform(df)
#   ndenum = np.ndenumerate(data)
#   # print(ndenum)
#   # print(len(ndenum))
#   # print(df.shape)
#   for (x,y), value in np.ndenumerate(data):
#     # print(x)
#     # print(y)
#     # print(df.columns[y])
#     # print(value)
#     oVal = df.at[x, df.columns[y]]
#     if (oVal == '' or pd.isnull(oVal)):
#       df.at[x, df.columns[y]] = value
#   return imputerFitted