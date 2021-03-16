import pandas as pd
import numpy as np
import json
import traceback

ROOT = 'project2_mldlivst/data'

pickMin = 2
pickProbMin = 0.7

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
    }
    tempRisePicks = []
    risePicksCount = 0
    tempDropPicks = []
    dropPicksCount = 0
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
          predFlag = row[predKey][0]
          predInt = int(row[predKey][1:])
          # pick rise
          if (predFlag == 'i' and predInt >= pickMin):
            # pass rise
            # rawPred? No, there won't be rawPred as any pred is already ctg processed classes
            pickItem = {
              "item": yName,
              "prediction": row[predKey],
              "predInt": predInt,
              "probability": prob
            }
            tempRisePicks.append(pickItem)
            risePicksCount += 1
          # pick drop
          if (predFlag == 'd' and predInt >= pickMin):
            # pass drop
            # rawPred? No, there won't be rawPred as any pred is already ctg processed classes
            pickItem = {
              "item": yName,
              "prediction": row[predKey],
              "predInt": predInt,
              "probability": prob
            }
            tempDropPicks.append(pickItem)
            dropPicksCount += 1
      except Exception as e:
        print('Exception in picking for ' + yName)
        # print(e)
        # print(traceback.format_exc())
    # order the picks
    item['risePicksCount'] = risePicksCount
    item['dropPicksCount'] = dropPicksCount
    item['risePicks'] = sorted(tempRisePicks, key=lambda item: item["predInt"] + item["probability"], reverse=True)
    item['dropPicks'] = sorted(tempDropPicks, key=lambda item: item["predInt"] + item["probability"], reverse=True)
    res.append(item)
  
  with open(ROOT + '/results/all_predictions_analysis.json', 'w') as fout:
    json.dump(res , fout, indent=2)