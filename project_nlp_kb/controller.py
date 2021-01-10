import pandas as pd
from sklearn.linear_model import LinearRegression
from . import preprocess
import general_preprocess as gPre

ROOT = 'project_nlp_kb'
X_CSV_FILEPATH = ROOT + '/data/articles1_PARSE_TREE_DICT.csv'
OUT_CSV_FILEPATH = ROOT + '/data/results/articles1_PARSE_TREE_DICT_CTG.csv'
YS_CSV_FILEPATH = ROOT + '/data/training1_Y.csv'
TRAINING_RESULTS_FILEPATH = ROOT + '/data/results/results.json'

linear_regression_results = []

def run():
  df = pd.read_csv(X_CSV_FILEPATH)
  # df = pd.read_csv(X_CSV_FILEPATH, error_bad_lines=False)
  # df = pd.read_csv(X_CSV_FILEPATH, sep=',')

  # preprocess
  # words vect by encode ctg
  for col in df:
    gPre.encodeCtgs(df, col)

  # write res
  df.to_csv(OUT_CSV_FILEPATH, index=False)


def train(stock, x_df, y_df):
  # check nan
  print(x_df.isnull().values.any())
  print(y_df.isnull().values.any())

  # LinearRegression
  x = x_df[:].values
  y = y_df[:].values
  reg = LinearRegression().fit(x, y)
  res = {}
  res['stock'] = stock
  res['score'] = reg.score(x, y)
  res['params'] = reg.get_params()
  # res['coef_'] = reg.coef_
  linear_regression_results.append(res)

    