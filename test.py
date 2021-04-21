import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# df test

# create df
testCols = ['c1', 'c2']
df = pd.DataFrame(columns = testCols)
print(df)

# append df
df = df.append({'c1': 1, 'c2': 2}, ignore_index=True)
df = df.append({'c1': 3, 'c2': 4}, ignore_index=True)
df = df.append({'c2': 6}, ignore_index=True)
print(df)

# create df2 with index
df2 = pd.DataFrame(index = df.index, columns = testCols)
print(df2)
df2 = df2.append({'c2': 6}, ignore_index=True)
print(df2)

# replace row at index
df2.loc[1] = {'c1': 3, 'c2': 4}
print(df2)

# replace cell
print(df2.loc[1]['c1'])
df2.at[2, 'c1'] = 11
print(df2)

# create df with np 2d arr
cols2 = ['a','s','d','f','g']
arr2d = np.zeros((3,5))
df3 = pd.DataFrame(arr2d, columns=cols2)
print(df3)

# join horizontally by index
dfJ = pd.concat([df, df2], axis=1)
print(dfJ)

# sort columns
df3 = df3.reindex(sorted(df3.columns), axis=1)
print(df3)


# test LogisticRegression
train_x = [[1,2,3],[4,5,6],[7,8,9]]
train_y1 = [[1,0,0],[0,1,0],[0,0,1]]
train_y2 = ['a','b','c']
train_y3 = [11,22,33]

x_df = pd.DataFrame(train_x)
y_df = pd.DataFrame(train_y3)

print(x_df)
print(y_df)
model = LogisticRegression().fit(x_df, y_df)
train_score = model.score(x_df, y_df)
prediction = model.predict(x_df)
probas = model.predict_proba(x_df)
print(train_score)
print(prediction)
print(probas)