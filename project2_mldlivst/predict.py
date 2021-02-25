from joblib import dump, load

ROOT = 'project2_mldlivst/data'

def run():
  x_df = pd.read_csv(ROOT + '/predict_X.csv')
  # load preprocessors
  x_imputer = load(ROOT + '/results/preprocess/x_imputer.joblib')
  x_scaler = load(ROOT + '/results/preprocess/x_scaler.joblib')
  # preprocess using the same process as in preprocess
  gPre.impute(x_df, imputer=x_imputer)
  gPre.fillMissingFinal(x_df, value=0)
  gPre.scaleNoFit(x_df, x_scaler)

  # load models
  