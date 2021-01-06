import pandas as pd
from sklearn.linear_model import LinearRegression
from . import preprocess

ROOT = 'project2_mldlivst'
X_CSV_FILEPATH = ROOT + '/data/training1_X.csv'
YS_CSV_FILEPATH = ROOT + '/data/training1_Y.csv'
TRAINING_RESULTS_FILEPATH = ROOT + '/data/results/results.json'

linear_regression_results = []

def run():
  # # v2 flow
  # train_df: temp df storing joined x_df and picked y
  # read csv to df
  #   x_df: X df
  #   ys_df: poss Ys df
  x_df = pd.read_csv(X_CSV_FILEPATH)
  ys_df = pd.read_csv(YS_CSV_FILEPATH)
  removeDates(x_df)
  removeDates(ys_df)

  # preprocess
  #   standardize scale
  #     x_df
  #     returned scaler keep for prediction use
  #     no scaling for y
  x_df_pre = processX(x_df)

  # train
  #   foreach col in ys_df:
  #     the col is y picked
  #     train_df = join x_df and picked col
  #     df of x and a y picked
  #     try diff MLs on the df
  for col in ys_df:
    # train_df = pd.concat([df1, df2], axis=1)
    if (col != 'datetime' and col != 'dateObj'):
      print('Using ' + str(col) + ' as Y.')
      # train_df = x_df.copy()
      # train_df['Y'] = ys_df[col]
      train(str(col), x_df, ys_df[col])

  # save results
  linear_regression_results_df = pd.DataFrame(linear_regression_results)
  df.to_json(TRAINING_RESULTS_FILEPATH, orient='records', lines=True)

def removeDates(df):
  del df['datetime']
  del df['dateObj']

def processX(df):
  # do all preprocess, feature extraction, ...
  res = preprocess.run(df)
  return res

def train(stock, x_df, y_df):
  # LinearRegression
  x = x_df
  y = y_df
  reg = LinearRegression().fit(x, y)
  res = {}
  res['stock'] = stock
  res['score'] = reg.score(x, y)
  res['params'] = reg.get_params()
  linear_regression_results.append(res)

    