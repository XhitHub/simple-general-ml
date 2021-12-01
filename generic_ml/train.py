import pandas as pd
import general_train as gTrain
from joblib import dump, load
from sklearn.linear_model import LinearRegression, LogisticRegression

trainCsv = 'generic_ml/data/train_preprocessed.csv'
modelFile = 'generic_ml/data/trained_model/model.joblib'

def createModel():
    model = LogisticRegression(max_iter=1000)
    return model

def run():
    trainDf = pd.read_csv(trainCsv)
