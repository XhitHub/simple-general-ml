import general_preprocess as gPre
from joblib import dump, load

ROOT = 'project2_mldlivst/data'
trimCount = 8

def run():
  x_df_all = pd.read_csv(ROOT + '/training1_X.csv')
  ys_df_all = pd.read_csv(ROOT + '/training1_Y.csv')
  dfLen = len(x_df_all.index)
  removeDates(x_df_all)
  removeDates(ys_df_all)
  # trim bad rows
  x_df = x_df_all[trimCount : dfLen-trimCount]
  ys_df = ys_df_all[trimCount : dfLen-trimCount]
  # preprocess X
  x_imputer = gPre.impute(x_df)
  x_scaler = gPre.scale(x_df)
  # preprocess Ys
  ys_imputer = gPre.impute(ys_df)
  ys_df = gPre.myDiscretize(ys_df, 5)
  # persist preprocessed data
  x_df.to_csv(ROOT + '/training1_X_pre.csv')
  ys_df.to_csv(ROOT + '/training1_Ys_pre.csv')
  # persist preprocessors
  try:
    dump(x_imputer, ROOT + '/results/preprocess/x_imputer.joblib')
    dump(x_scaler, ROOT + '/results/preprocess/x_scaler.joblib')
  except Exception as e3:
    print('preprocess err pt 1: dump')
    print(e3)




def removeDates(df):
  del df['datetime']
  del df['dateObj']