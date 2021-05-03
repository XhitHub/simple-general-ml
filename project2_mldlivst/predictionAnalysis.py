import pandas as pd
import numpy as np
import json
import math
import traceback

ROOT = 'project2_mldlivst/data'

CLASS_GD = 2

max_count = 900

training_ratio = 0.7

pickMin = 5
# pickProbMin = 0.75
pickProbMin = 0.8
testScoreMin = 0.75
trustProbMin = 0.8

def run():
  df = pd.read_csv(ROOT + '/results/all_predictions.csv')
  dfTrainingResults = pd.read_csv(ROOT + '/results/training_results.csv')
  # refers to train Y csv to see wt Ys are available to be predicted
  train_ys_df = pd.read_csv(ROOT + '/preprocessed/training1_Y.csv')
  raw_train_ys_df = pd.read_csv(ROOT + '/validation1_Y.csv')
  # df.set_index('datetime', inplace=True)
  df.set_index('datetime', inplace=True)
  raw_train_ys_df.set_index('datetime', inplace=True)
  dfTrainingResults.set_index('stock', inplace=True)
  # del train_ys_df['datetime']
  yNames = train_ys_df.columns
  print(yNames)
  print(yNames[0])
  # return


  # simple joined prediction and validation df
  dfJoined = pd.concat([df, raw_train_ys_df], axis=1)
  dfJoined = dfJoined.reindex(sorted(dfJoined.columns), axis=1)
  dfJoined.to_csv(ROOT + '/results/all_predictions_analysis_LogisticRegV2Joined.csv')


  # analysis 2
  # class gd
  #   class 3
  # target analysis res
  #   cols
  #     predY
  #     model test score
  #     class gd count
  #     avg err of trusted class gd cells
  #       trusted class gd cells
  #         class gd cells with proba >= min trust threshold
  #       as in PROD, will ivst according to trusted class gd cells. this estimates how doing ivst according to trusted class gd cells will goes
  # impl
  #   in test split part:
  #     for each predY col
  #       1. count class gd
  #       2. pick class gd cells
  #         compare with df_valid
  #         calc avg err of them
  testCount = int(len(dfJoined.index) * (1 - training_ratio))
  dfJoinedTest = dfJoined[:testCount]
  # predYCols = raw_train_ys_df.columns.values
  predYCols = raw_train_ys_df.columns.values[:max_count]
  res2 = []
  count = 0
  for col in predYCols:
    count += 1
    # if (count > max_count):
    #   print('count exceeded max, should break')
    #   break
    print('analysis 2: ' + str(count) + '/' + str(len(predYCols)))
    actualYs = dfJoinedTest[col]
    preds = dfJoinedTest[col + '_predict']
    predPs = dfJoinedTest[col + '_predict_maxP']
    modelTestScore = dfTrainingResults.loc[col, 'test_score']
    # classGdCount = preds.count(CLASS_GD)
    totalPickedError = 0
    pickedCount = 0
    for i,pred in enumerate(preds):
      if pred == CLASS_GD and predPs[i] >= trustProbMin:
        # pick
        pickedCount += 1
        totalPickedError += actualYs[i]*100 - pred
    item = {
      "Y": col,
      "modelTestScore": modelTestScore,
      "pickedCount": pickedCount,
      "totalPickedError": totalPickedError,
    }
    if (pickedCount > 0):
      avgPickedErr = totalPickedError / pickedCount
      item['avgPickedErr'] = avgPickedErr
    res2.append(item)
  res2Df = pd.DataFrame(res2)
  res2Df.to_csv(ROOT + '/results/all_predictions_analysis_LogisticRegV2_analysis2.csv')


  
  # picking
  dfJoinedTest = dfJoined[:testCount]
  # pickRes = pick(dfJoinedTest, dfTrainingResults, train_ys_df.columns.values)
  processedTrainYCols = train_ys_df.columns.values[:max_count]
  pickRes = pick(dfJoinedTest, dfTrainingResults, processedTrainYCols)
  with open(ROOT + '/results/picks_Log_reg_v2.json', 'w') as fout:
    json.dump(pickRes , fout, indent=2)
  pickResDf = pd.DataFrame(pickRes['daysPicks'])
  pickResDf.to_csv(ROOT + '/results/picks_Log_reg_v2.csv')


def is_gd_class(c):
  return c == CLASS_GD

def pick(all_pred_valid_df, train_res_df, yCols, max_pick_count=7):
  # all_pred_valid_df: all_predictions.csv joined validation1_Y.csv
  # train_res_df: training_results.csv
  # yCols: all cols of train1_Y

  # determ cols with gd enough test score
  gdCols = []
  for col in yCols:
    testScore = train_res_df.loc[col, 'test_score']
    if (testScore >= testScoreMin):
      gdCols.append(col)
  res = []
  resWithAvg = []
  for i, row in all_pred_valid_df.iterrows():
    rowPicks = []
    rowPicksCount = 0
    totalActualY = 0
    totalErr = 0
    for col in gdCols:
      pred = row[col + '_predict']
      predP = row[col + '_predict_maxP']
      actualY = row[col] * 100
      predErr = actualY - pred
      if (is_gd_class(pred) and predP >= trustProbMin):
        item = {
          "pick": col,
          "pred": pred,
          "predP": predP,
          "actualY": actualY,
          "predErr": predErr,
        }
        rowPicks.append(item)
        rowPicksCount += 1
        totalActualY += actualY
        totalErr += predErr

    item = {
      'datetime': i,
      'initialPicksCount': rowPicksCount,
    }

    # pick with max pick count and non repeating stock, i.e. same stock XdR only pick once
    if (len(rowPicks) > 0):
      limitedTotalActualY = 0
      limitedTotalErr = 0
      rowPicks = sorted(rowPicks, key=lambda k: -k['predP'])
      limitedRowPicks = []
      pickedStocks = []
      for pick in rowPicks:
        if (len(limitedRowPicks) >= max_pick_count):
          break
        stock = pick['pick'][:9]
        if (stock not in pickedStocks):
          limitedRowPicks.append(pick)
          pickedStocks.append(stock)
          limitedTotalActualY += pick['actualY']
          limitedTotalErr += pick['predErr']
      limitedPicksCount = len(limitedRowPicks)
      item['limitedPicksCount'] = limitedPicksCount
      item['avgActualY'] = limitedTotalActualY / limitedPicksCount
      item['avgErr'] = limitedTotalErr / limitedPicksCount

      item['limitedPicks'] = limitedRowPicks
      item['limitedPicks_simple'] = [d['pick'] for d in limitedRowPicks]
    res.append(item)

  itemsWithAvg = [d for d in res if 'avgActualY' in d and not np.isnan(d['avgActualY'])]
  # print(itemsWithAvg)
  allDaysAvgActualY = sum(d['avgActualY'] for d in itemsWithAvg) / len(itemsWithAvg)

  bigRes = {
    'allDaysAvgActualY': allDaysAvgActualY,
    'daysPicks': res,
  }
  return bigRes
      

def pick_v1(all_pred_valid_df, train_res_df, yCols):
  # all_pred_valid_df: all_predictions.csv joined validation1_Y.csv
  # train_res_df: training_results.csv
  # yCols: all cols of train1_Y
  # determ cols with gd enough test score
  gdCols = []
  for col in yCols:
    testScore = train_res_df.loc[col, 'test_score']
    if (testScore >= testScoreMin):
      gdCols.append(col)
  res = []
  for i, row in all_pred_valid_df.iterrows():
    rowPicks = []
    rowPicksCount = 0
    totalActualY = 0
    totalErr = 0
    for col in gdCols:
      pred = row[col + '_predict']
      predP = row[col + '_predict_maxP']
      actualY = row[col] * 100
      predErr = actualY - pred
      if (pred == CLASS_GD and predP >= trustProbMin):
        item = {
          "pick": col,
          "pred": pred,
          "predP": predP,
          "actualY": actualY,
          "predErr": predErr,
        }
        rowPicks.append(item)
        rowPicksCount += 1
        totalActualY += actualY
        totalErr += predErr
    item = {
      'datetime': i,
      'picksCount': rowPicksCount,
    }
    if (rowPicksCount > 0):
      item['avgActualY'] = totalActualY / rowPicksCount
      item['avgErr'] = totalErr / rowPicksCount
    item['picks'] = rowPicks
    res.append(item)
  return res


  # res = []
  # resErrorList = []
  # total = len(df.index)
  # for index, row in df.iterrows():
  #   print('Prediction analysis: ('+str(index)+'/'+str(total)+')')
  #   rowDateTime = row['datetime']
  #   item = {
  #     "datetime": rowDateTime,
  #   }
  #   tempRisePicks = []
  #   risePicksCount = 0
  #   tempDropPicks = []
  #   dropPicksCount = 0
  #   totalRisePicksError = 0
  #   for i in range(0,len(yNames)):
  #     try:
  #       yName = yNames[i]
  #       predKey = yName + '_predict'
  #       item[predKey] = row[predKey]
  #       if (rowDateTime in raw_train_ys_df.index):
  #         predError = raw_train_ys_df.loc[rowDateTime][yName] - row[predKey]
  #         item[yName + '_predError'] = predError
  #     except Exception as e:
  #       print('Predict Analysis ' + yName + ' err pt1: ')
  #       print(e)
  #   res.append(item)
  
  # with open(ROOT + '/results/all_predictions_analysis_SVR.json', 'w') as fout:
  #   json.dump(res , fout, indent=2)

  # resDf = pd.DataFrame(res)
  # resDf.to_csv(ROOT + '/results/all_predictions_analysis_SVR.csv')


  # resErrorListDf = pd.DataFrame(resErrorList)
  # resErrorListDf.to_csv(ROOT + '/results/all_predictions_analysis_avg_errors.csv', index=False, sep="\t")