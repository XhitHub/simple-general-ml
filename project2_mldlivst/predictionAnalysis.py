import pandas as pd
import numpy as np
import json
import math
import traceback

ROOT = 'project2_mldlivst/data'

pickMin = 5
# pickProbMin = 0.75
pickProbMin = 0.8

def run():
  df = pd.read_csv(ROOT + '/results/all_predictions.csv')
  # refers to train Y csv to see wt Ys are available to be predicted
  train_ys_df = pd.read_csv(ROOT + '/preprocessed/training1_Y.csv')
  raw_train_ys_df = pd.read_csv(ROOT + '/validation1_Y.csv')
  df.set_index('datetime', inplace=True)
  raw_train_ys_df.set_index('datetime', inplace=True)
  # del train_ys_df['datetime']
  yNames = train_ys_df.columns
  print(yNames)
  print(yNames[0])
  # return

  res = []
  resErrorList = []
  for index, row in df.iterrows():
    rowDateTime = row['datetime']
    item = {
      "datetime": rowDateTime,
    }
    tempRisePicks = []
    risePicksCount = 0
    tempDropPicks = []
    dropPicksCount = 0
    totalRisePicksError = 0
    for i in range(0,len(yNames)):
      try:
        yName = yNames[i]
        predKey = yName + '_predict'

        predError = row[predKey] - raw_train_ys_df.loc[rowDateTime][yName]
        item[yName + '_predError'] = predError
      except Exception as e:
        print('Predict Analysis ' + yName + ' err pt1: ')
        print(e)
    res.append(item)
  
  with open(ROOT + '/results/all_predictions_analysis_SVR.json', 'w') as fout:
    json.dump(res , fout, indent=2)

  resDf = pd.DataFrame(res)
  resDf.to_csv(ROOT + '/results/all_predictions_analysis_SVR.csv')


  resErrorListDf = pd.DataFrame(resErrorList)
  resErrorListDf.to_csv(ROOT + '/results/all_predictions_analysis_avg_errors.csv', index=False, sep="\t")