import pandas as pd
import general_preprocess as gPre
from joblib import dump, load

ROOT = 'project2_mldlivst/data'
trimCount = 8
yCtgArr = [2,5]

def run():
  x_df_all = pd.read_csv(ROOT + '/training1_X.csv')
  ys_df_all = pd.read_csv(ROOT + '/training1_Y.csv')
  dfLen = len(x_df_all.index)
  removeDates(x_df_all)
  removeDates(ys_df_all)
  # preprocess X
  x_imputer = gPre.impute(x_df_all)
  dump(x_imputer, ROOT + '/results/preprocess/x_imputer.joblib')
  gPre.fillMissingFinal(x_df_all, value=0)
  x_scaler = gPre.scale(x_df_all)
  # preprocess Ys
  ys_imputer = gPre.impute(ys_df_all)
  dump(ys_imputer, ROOT + '/results/preprocess/ys_imputer.joblib')
  gPre.fillMissingFinal(ys_df_all, value=0)
  # ys_df_all = gPre.myDiscretize(ys_df_all, 5)
  ys_df_all = gPre.percentageChangeToCtg(ys_df_all, yCtgArr)
  # trim bad rows
  x_df = x_df_all[trimCount : dfLen-trimCount]
  ys_df = ys_df_all[trimCount : dfLen-trimCount]
  # save preprocessed data
  x_df.to_csv(ROOT + '/preprocessed/training1_X.csv', index=False)
  ys_df.to_csv(ROOT + '/preprocessed/training1_Y.csv', index=False)
  # persist preprocessors
  try:
    # dump(x_imputer, ROOT + '/results/preprocess/x_imputer.joblib')
    # dump(ys_imputer, ROOT + '/results/preprocess/ys_imputer.joblib')
    dump(x_scaler, ROOT + '/results/preprocess/x_scaler.joblib')
  except Exception as e3:
    print('preprocess err pt 1: dump')
    print(e3)



def removeDates(df):
  if ('datetime' in df.columns):
    del df['datetime']
  if ('dateObj' in df.columns):
    del df['dateObj']