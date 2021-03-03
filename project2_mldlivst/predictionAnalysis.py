import pandas as pd
import numpy as np
import json
import traceback

ROOT = 'project2_mldlivst/data'

pickPredMin = 3
pickProbMin = 0.8

def run():
  df = pd.read_csv(ROOT + '/results/all_predictions.csv')
  # refers to train Y csv to see wt Ys are available to be predicted
  train_ys_df = pd.read_csv(ROOT + '/preprocessed/training1_Y.csv')
  # del train_ys_df['datetime']
  yNames = train_ys_df.columns
  print(yNames)
  print(yNames[0])
  # return

  res = []
  for index, row in df.iterrows():
    item = {
      "datetime": row['datetime'],
      "picks": []
    }
    for i in range(0,len(yNames)):
      try:
        yName = yNames[i]
        predKey = yName + '_predict'
        predProbKey = yName + '_predict_proba_max'
        # check if the yName's prediction qualifies as picks
        # print(row)
        prob = row[predProbKey]
        # prob = row['0020.XHKG_closeDelta1dR_predict_proba_max']
        # print(prob)
        if (prob >= pickProbMin):
          pred = int(row[predKey][0])
          if (pred >= pickPredMin):
            pickItem = {
              "item": yName,
              "prediction": row[predKey],
              "probability": prob
            }
            item['picks'].append(pickItem)
      except Exception as e:
        print('Exception in picking for ' + yName)
        # print(e)
        # print(traceback.format_exc())
    res.append(item)
  
  with open(ROOT + '/results/all_predictions_analysis.json', 'w') as fout:
    json.dump(res , fout)