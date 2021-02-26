from joblib import dump, load
import pandas as pd
import general_preprocess as gPre

ROOT = 'project2_mldlivst/data'
maxCount = 500

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
  for col in train_ys_df:
    # col is the Y name
    try:
      if (col != 'datetime' and col != 'dateObj'):
        continue
      if (count == total):
        break
      count += 1
      yName = str(col)
      print('Predicting ' + yName + '. ('+str(count)+'/'+str(total)+')')
      resDf = pd.DataFrame(data=[x_dfDatetime], columns=['datatime'])
      modelPath = ROOT + '/results/models/' + yName + '.joblib'
      model = load(modelPath)
      pred = model.predict(x_df)
      resDf['pred'] = pred
      predProbs = model.predict_proba(x_df)
      for i in len(predProbs):
        yClass = model.classes_[i]
        # label = str(col) + '_' + str(yClass)
        label = str(yClass)
        resDf[label] = predProbs[i]
      resDf.to_csv(ROOT + '/prediction/' + yName + '.csv', index=False)
    except Exception as e:
      print('Predict err: ' + yName)
      print(e)

def removeDates(df):
  del df['datetime']
  del df['dateObj']
