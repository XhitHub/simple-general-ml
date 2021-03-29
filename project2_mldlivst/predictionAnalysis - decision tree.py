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
            ### pass rise
            # rawPred? No, there won't be rawPred as any pred is already ctg processed classes
            pickItem = {
              "item": yName,
              "prediction": row[predKey],
              "predInt": predInt,
              "probability": prob
            }
            # actual result, available for old enough records
            try:
              actualRow = raw_train_ys_df.loc[rowDateTime]
              actualRes = actualRow[yName]
              if (math.isnan(actualRes)):
                actualRes = 'nan'
                error = 'nan'
              else:
                # to percentage
                actualRes *= 100
                error = actualRes - predInt
                totalRisePicksError += error
              pickItem['actual_result'] = actualRes
              pickItem['error'] = error
            except Exception as e:
              print('Exception in adding actual result ' + rowDateTime + ', ' + yName)
              print(traceback.format_exc())
            tempRisePicks.append(pickItem)
            risePicksCount += 1
          # pick drop
          if (predFlag == 'd' and predInt >= pickMin):
            ### pass drop
            # rawPred? No, there won't be rawPred as any pred is already ctg processed classes
            pickItem = {
              "item": yName,
              "prediction": row[predKey],
              "predInt": predInt,
              "probability": prob
            }
            # actual result, available for old enough records
            try:
              actualRow = raw_train_ys_df.loc[rowDateTime]
              actualRes = actualRow[yName]
              if (math.isnan(actualRes)):
                actualRes = 'nan'
              pickItem['actual_result'] = actualRes
            except Exception as e:
              print('Exception in adding actual result ' + rowDateTime + ', ' + yName)
              print(traceback.format_exc())
            tempDropPicks.append(pickItem)
            dropPicksCount += 1
      except Exception as e:
        print('Exception in picking for ' + yName)
        # print(e)
        # print(traceback.format_exc())
    # order the picks
    item['risePicksCount'] = risePicksCount
    item['dropPicksCount'] = dropPicksCount
    if (risePicksCount > 0 and totalRisePicksError != 0):
      avgRisePicksError = totalRisePicksError / risePicksCount
      item['avgRisePicksError'] = avgRisePicksError
      resErrorListItem = {
        "datetime": rowDateTime,
        "avgRisePicksError": avgRisePicksError,
        "risePicksCount": risePicksCount,
      }
      resErrorList.append(resErrorListItem)
    item['risePicks'] = sorted(tempRisePicks, key=lambda item: item["predInt"] + item["probability"], reverse=True)
    item['dropPicks'] = sorted(tempDropPicks, key=lambda item: item["predInt"] + item["probability"], reverse=True)
    res.append(item)
  
  with open(ROOT + '/results/all_predictions_analysis.json', 'w') as fout:
    json.dump(res , fout, indent=2)

  resErrorListDf = pd.DataFrame(resErrorList)
  resErrorListDf.to_csv(ROOT + '/results/all_predictions_analysis_avg_errors.csv', index=False, sep="\t")