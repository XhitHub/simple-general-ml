from joblib import dump, load
import pandas as pd
import numpy as np
import general_preprocess as gPre

ROOT = 'project2_mldlivst/data'
maxCount = 900

def run():
  x_df = pd.read_csv(ROOT + '/predict_X.csv')
  x_dfDatetime = x_df['datetime']
  removeDates(x_df)
  # load preprocessors
  x_imputer = load(ROOT + '/results/preprocess/x_imputer.joblib')
  # x_imputer = load(ROOT + '/results/preprocess/ys_imputer.joblib')
  x_scaler = load(ROOT + '/results/preprocess/x_scaler.joblib')
  # preprocess using the same process as in preprocess
  gPre.imputeNoFit(x_df, imputer=x_imputer)
  gPre.fillMissingFinal(x_df, value=0)
  gPre.scaleNoFit(x_df, x_scaler)

  # load models
  # stocks = pd.read_csv(ROOT + '/symbols/marketstack.csv')
  # for stock in stocks:
  #   symb = stock['Symbol']

  # refers to train Y csv to see wt Ys are available to be predicted
  count = 0
  train_ys_df = pd.read_csv(ROOT + '/preprocessed/training1_Y.csv')
  total = maxCount
  allResDf = pd.DataFrame()
  allResDf['datetime'] = x_dfDatetime
  # print(len(train_ys_df))
  for col in train_ys_df:
    # col is the Y name
    try:
      if (col == 'datetime' or col == 'dateObj'):
        continue
      if (count == total):
        break
      count += 1
      yName = str(col)
      print('Predicting ' + yName + '. ('+str(count)+'/'+str(total)+')')
      resDf = pd.DataFrame()
      resDf['datetime'] = x_dfDatetime
      modelPath = ROOT + '/results/models/' + yName + '.joblib'
      model = load(modelPath)
      pred = model.predict(x_df)
      predP = model.predict_proba(x_df)
      predPMaxs = [max(probaRow) for probaRow in predP]
      resDf['predict'] = pred
      allResDf[yName + '_predict'] = pred
      resDf['predict_maxP'] = predPMaxs
      allResDf[yName + '_predict_maxP'] = predPMaxs
    except Exception as e:
      print('Predict ' + yName + ' err pt1: ')
      print(e)
    # try:
    #   predProbs = model.predict_proba(x_df)
    #   pMax = np.max(predProbs, axis=1)
    #   resDf['predict_proba_max'] = pMax
    #   allResDf[yName + '_predict_proba_max'] = pMax
    # except Exception as e:
    #   print('Predict ' + yName + ' err pt2: ')
    #   print(e)
    try:
      resDf.to_csv(ROOT + '/prediction/' + yName + '.csv', index=False)
    except Exception as e:
      print('Predict ' + yName + ' err pt3: ')
      print(e)
  try:
    allResDf.to_csv(ROOT + '/results/all_predictions.csv', index=False)
  except Exception as e:
    print('pt4: Write allResDf err: ')
    print(e)

def removeDates(df):
  if ('datetime' in df.columns):
    del df['datetime']
  if ('dateObj' in df.columns):
    del df['dateObj']



      # wrong proba approach
      # print(predProbs)
      # for i in range(0,len(predProbs)):
      #   yClass = model.classes_[i]
      #   # label = str(col) + '_' + str(yClass)
      #   label = str(yClass)
      #   resDf[label] = predProbs[i]