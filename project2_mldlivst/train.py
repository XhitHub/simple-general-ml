import math
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from joblib import dump, load

# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Reshape
# from keras.layers import Flatten
# from keras.layers import Conv2D
# from keras.layers import Conv2DTranspose
# from keras.layers import LeakyReLU
# from keras.layers import Dropout

from . import preprocess
import general_preprocess as gPre
import general_train as gTrain

ROOT = 'project2_mldlivst/data'
# to try on some of the data first, set maxCount
maxCount = 900

xAttrCount = 2400
# min_samples_leaf = 0.15
min_samples_leaf = 0.2 #gd
# min_samples_leaf = 0.4
# min_samples_leaf = .5

training_ratio = 0.7

training_results = []

def makeModel():
  model = LogisticRegression(max_iter=1000)
  return model

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
  training_results_df.to_json(ROOT + '/results/training_results.json', orient='records', lines=True)
  training_results_df.to_csv(ROOT + '/results/training_results.csv')

def getModelName(stock):
  modelName = stock.replace('.','_')
  return modelName

# def defineModel():
#   c = xAttrCount
#   model = Sequential()
#   for i in range(0,9):
#     c = int(c/2)
#     model.add(Dense(c, input_shape=(xAttrCount,), activation='tanh'))
#   model.add(Dense(1, activation='sigmoid'))
#   # compile model
#   opt = Adam(lr=0.0002, beta_1=0.5)
#   model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#   return model

def train(stock, x_df, y_df):
  split_train_test = False

  # LinearRegression
  x = x_df.values
  y = y_df.values

  print(stock)

  # no split
  x_train = x
  y_train = y
  x_test = None
  y_test = None

  # # split train test set v1
  # split_train_test = True
  # tc = int(len(x_df.index) * training_ratio)
  # print(tc)
  # x_train = x[:tc]
  # y_train = y[:tc]
  # x_test = x[tc:]
  # y_test = y[tc:]

  # split train test set v2
  split_train_test = True
  tc = int(len(x_df.index) * (1-training_ratio))
  print(tc)
  x_test = x[:tc]
  y_test = y[:tc]
  x_train = x[tc:]
  y_train = y[tc:]

  res = {}
  res['stock'] = stock
  try:
    model = makeModel().fit(x_train, y_train)
    # model = LogisticRegression().fit(x_train, y_train)
    # model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf).fit(x_train, y_train)
    res['train_score'] = model.score(x_train, y_train)
    if (split_train_test):
      res['test_score'] = model.score(x_test, y_test)
    # res['params'] = model.get_params()
  except Exception as e:
    print('train err pt 1')
    print(e)

  # persist model
  try:
    mn = stock
    dump(model, ROOT + '/results/models/' + mn + '.joblib')
  except Exception as e3:
    print('train err pt 3: dump')
    print(e3)
    
  training_results.append(res)