import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from . import preprocess
import general_preprocess as gPre

ROOT = 'project2_mldlivst'
X_CSV_FILEPATH = ROOT + '/data/training1_X.csv'
X_CSV_FILEPATH_TEST = ROOT + '/data/test/training1_X.csv'
YS_CSV_FILEPATH = ROOT + '/data/training1_Y.csv'
TRAINING_RESULTS_FILEPATH = ROOT + '/data/results/results.json'

linear_regression_results = []

def run():
  # # v2 flow
  # train_df: temp df storing joined x_df and picked y
  # read csv to df
  #   x_df: X df
  #   ys_df: poss Ys df
  x_df_all = pd.read_csv(X_CSV_FILEPATH)
  ys_df_all = pd.read_csv(YS_CSV_FILEPATH)
  trimCount = 8
  dfLen = len(x_df_all.index)
  removeDates(x_df_all)
  removeDates(ys_df_all)

  # preprocess
  #   standardize scale
  #     x_df
  #     returned scaler keep for prediction use
  #     no scaling for y

  # scalers, ... obtained in preo=process
  preprocess_x_res = processX(x_df_all)
  preprocess_y_res = processX(ys_df_all)
  # gPre.impute(ys_df_all)

  x_df = x_df_all[trimCount : dfLen-trimCount]
  ys_df = ys_df_all[trimCount : dfLen-trimCount]

  # inspect x_df
  x_df.to_csv(ROOT + '/data/test/x_t1_imputed.csv')
  ys_df.to_csv(ROOT + '/data/test/ys_t1_imputed.csv')

  # train
  #   foreach col in ys_df:
  #     the col is y picked
  #     train_df = join x_df and picked col
  #     df of x and a y picked
  #     try diff MLs on the df
  maxCount = 200
  count = 0
  for col in ys_df:
    # train_df = pd.concat([df1, df2], axis=1)
    try:
      if (col != 'datetime' and col != 'dateObj'):
        print('Using ' + str(col) + ' as Y.')
        # train_df = x_df.copy()
        # train_df['Y'] = ys_df[col]
        train(str(col), x_df, ys_df[col])
        count += 1
    except Exception as e: 
      print(e)
    if count == maxCount:
      break

  # save results
  linear_regression_results_df = pd.DataFrame(linear_regression_results)
  linear_regression_results_df.to_json(TRAINING_RESULTS_FILEPATH, orient='records', lines=True)

def removeDates(df):
  del df['datetime']
  del df['dateObj']

def processX(df):
  # do all preprocess, feature extraction, ...
  res = preprocess.run(df)
  return res

def processY(df):
  # do all preprocess, feature extraction, ...
  res = gPre.impute(df)
  return res

def train(stock, x_df, y_df):
  # check nan
  # print(x_df.isnull().values.any())
  # print(y_df.isnull().values.any())

  # LinearRegression
  x = x_df.values
  y = y_df.values

  print(stock)
  # print(x.shape)
  # print(y.shape)
  # x = x_df[:].values
  # y = y_df[:].values
  tc = 180
  x_train = x[:tc]
  y_train = y[:tc]
  x_test = x[tc:]
  y_test = y[tc:]
  try:
    reg = LinearRegression().fit(x_train, y_train)
  except Exception as e:
    print(e)
  res = {}
  res['stock'] = stock
  res['score'] = reg.score(x_test, y_test)
  res['params'] = reg.get_params()

  # test prediction
  pY = reg.predict(x)
  print(pY.shape)
  py_df = pd.DataFrame(pY)
  pred_df = pd.concat([y_df, py_df], axis=1)
  pred_df.to_csv(ROOT + '/data/test_pred/'+stock[0:4]+'.csv')

  # custom score
  sum = 0
  for i,predY in enumerate(pY):
    diff = predY - y[i]
    # print(diff)
    sum += (diff * diff)
  print(sum)
  res['custom_training_error_sum'] = sum
  res['custom_training_error'] = math.sqrt(sum / len(pY))


  # res['coef_'] = reg.coef_
  linear_regression_results.append(res)

    