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
  resPath = ROOT + '/data/preprocessed.csv'
  df.to_csv(resPath, index=False)

def processX(df):
  # do all preprocess, feature extraction, ...
  res = preprocess.run(df)
  return res

# # v2 flow
# train_df: temp df storing joined x_df and picked y
# read csv to df
#   x_df: X df
#   ys_df: poss Ys df
# preprocess
#   standardize scale
#     x_df
#     returned scaler keep for prediction use
#     no scaling for y
# train
#   foreach col in ys_df:
#     the col is y picked
#     train_df = join x_df and picked col
#     df of x and a y picked
#     try diff MLs on the df
    