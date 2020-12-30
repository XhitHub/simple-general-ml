import pandas as pd
from . import preprocess

ROOT = 'project2_mldlivst'

def run():
  df = pd.read_csv(ROOT + '/data/raw.csv')
  print('Before process X:')
  print(df)
  res = processX(df)
  print('After process X:')
  print(df)
  print(res)
  resPath = ROOT + '/data/processed.csv'
  df.to_csv(resPath, index=False)

def processX(df):
  # do all preprocess, feature extraction, ...
  res = preprocess.run(df)
  return res
