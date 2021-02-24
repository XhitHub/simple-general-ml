import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from joblib import dump, load
from . import preprocess
import general_preprocess as gPre
import general_train as gTrain

ROOT = 'project2_mldlivst/data'
# to try on some of the data first, set maxCount
maxCount = 500

training_results = []

def run():
  count = 0
  x_df = pd.read_csv(ROOT + '/preprocessed/training1_X.csv')
  ys_df = pd.read_csv(ROOT + '/preprocessed/training1_Y.csv')
  total = maxCount
  for col in ys_df:
    # col is the stock
    try:
      if (col != 'datetime' and col != 'dateObj'):
        print('Training with ' + str(col) + ' as Y. ('+str(count)+'/'+str(total)+')')
        # train_df = x_df.copy()
        # train_df['Y'] = ys_df[col]
        train(str(col), x_df, ys_df[col])
        count += 1
    except Exception as e: 
      print(e)
    if count == maxCount:
      break
  # save results
  training_results_df = pd.DataFrame(training_results)
  training_results_df.to_json(ROOT + '/results/results.json', orient='records', lines=True)

def getModelName(stock):
  modelName = stock.replace('.','_')
  return modelName

def train(stock, x_df, y_df):
  # LinearRegression
  x = x_df.values
  y = y_df.values

  print(stock)

  # no split
  x_train = x
  y_train = y
  x_test = None
  y_test = None

  # # split train test set
  # tc = 180
  # x_train = x[:tc]
  # y_train = y[:tc]
  # x_test = x[tc:]
  # y_test = y[tc:]

  res = {}
  res['stock'] = stock
  try:
    # model = LinearRegression().fit(x_train, y_train)
    model = DecisionTreeClassifier(min_samples_leaf=0.15).fit(x_train, y_train)
    res['train_score'] = model.score(x_train, y_train)
    if (x_test != None and y_test != None):
      res['test_score'] = model.score(x_test, y_test)
    # res['params'] = model.get_params()
  except Exception as e:
    print('train err pt 1')
    print(e)

  # custom eval decision tree
  try:
    treeEvalRes = gTrain.eval_decision_tree(model)
    res['tree_eval'] = treeEvalRes
  except Exception as e2:
    print('train err pt 2')
    print(e2)

  # persist model
  try:
    mn = stock
    dump(model, ROOT + '/results/models/' + mn + '.joblib')
  except Exception as e3:
    print('train err pt 3: dump')
    print(e3)
    
  training_results.append(res)