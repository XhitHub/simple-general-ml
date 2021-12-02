import pandas as pd
import general_preprocess as gPre
import general_train as gTrain
from joblib import dump, load
from sklearn.linear_model import LinearRegression, LogisticRegression

rawTrainCsv = 'generic_ml/data/train.csv'
trainCsv = 'generic_ml/data/train_preprocessed.csv'
predictCsv = 'generic_ml/data/predict.csv'
predictionResCsv = 'generic_ml/data/predict_res.csv'
predictionAnalysisCsv = 'generic_ml/data/predict_analysis.csv'
imputerFile = 'generic_ml/data/preprocessor/imputer.joblib'
scalerFile = 'generic_ml/data/preprocessor/scaler.joblib'
modelFile = 'generic_ml/data/trained_model/model.joblib'
yName = 'close_compare_+10days_ctg'

def createModel():
    model = LogisticRegression(max_iter=1000)
    return model

def preprocess():
  # preprocess
  trainDf = pd.read_csv(rawTrainCsv)
  imputer = gPre.impute(trainDf)
  dump(imputer, imputerFile)
  gPre.fillMissingFinal(trainDf, value=0)
  scaler = gPre.scale(trainDf)
  dump(scaler, scalerFile)
  trainDf.to_csv(resTrainCsv)

def train():
    allDf = pd.read_csv(trainCsv)
    trainDf = allDf
    xDf = trainDf.drop([yName], axis=1)
    yDf = trainDf[[yName]]
    model = createModel().fit(xDf.values, yDf.values)
    res = {}
    res['train_score'] = model.score(xDf.values, yDf.values)
    dump(model, modelFile)
    return res

def predict():
    x_df = pd.read_csv(predictCsv)
    imputer = load(imputerFile)
    scaler = load(scalerFile)
    gPre.imputeNoFit(x_df, imputer=imputer)
    gPre.fillMissingFinal(x_df, value=0)
    gPre.scaleNoFit(x_df, scaler)
    model = load(modelFile)
    pred = model.predict(x_df)
    predP = model.predict_proba(x_df)
    resDf = pd.DataFrame()
    resDf['pred'] = pred
    resDf['predP'] = predP
    resDf.to_csv(predictionResCsv)

def analyze():
    predResDf = pd.read_csv(predictionResCsv)
    actualYDf = pd.read_csv(trainCsv)
    predResDf['pred_diff'] = actualYDf[yName] - predResDf['pred']
    predResDf.to_csv(predictionAnalysisCsv)