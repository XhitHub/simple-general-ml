import pandas as pd
import general_preprocess as gPre
from joblib import dump, load

trainCsv = 'generic_ml/data/train.csv'
resTrainCsv = 'generic_ml/data/train_preprocessed.csv'
imputerFile = 'generic_ml/data/preprocessor/imputer.joblib'
scalerFile = 'generic_ml/data/preprocessor/scaler.joblib'

def run():
  # preprocess
  trainDf = pd.read_csv(trainCsv)
  imputer = gPre.impute(trainDf)
  dump(imputer, imputerFile)
  gPre.fillMissingFinal(trainDf, value=0)
  scaler = gPre.scale(trainDf)
  dump(scaler, scalerFile)
  trainDf.to_csv(resTrainCsv)