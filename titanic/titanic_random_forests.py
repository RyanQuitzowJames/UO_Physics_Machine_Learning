import pandas as pd
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv', header=0)

df['Gender'] = df.Sex.map({'female': 0, 'male': 1}).astype(int)
df_test['Gender'] = df_test.Sex.map({'female': 0, 'male': 1}).astype(int)

median_ages = [
    [
        df[(df.Gender == i) & (df.Pclass == j+1)].Age.dropna().median()
    for j in range(3)]
for i in range(2)]

median_ages = np.array(median_ages)

df['AgeFill'] = df.Age
df_test['AgeFill'] = df_test.Age

median_fares = [
    [
        df[(df.Gender == i) & (df.Pclass == j+1)].Fare.dropna().median()
    for j in range(3)]
for i in range(2)]

median_fares = np.array(median_ages)

df['FareFill'] = df.Fare
df_test['FareFill'] = df_test.Fare

for i in range(2):
    for j in range(3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)

for i in range(2):
    for j in range(3):
        df.loc[(df.Fare.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'FareFill'] = median_fares[i, j]
        df_test.loc[(df_test.Fare.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1), 'FareFill'] = median_fares[i, j]

df['FareIsNull'] = pd.isnull(df.Fare).astype(int)
df_test['FareIsNull'] = pd.isnull(df_test.Fare).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass

town_to_number = {'C': 0, 'S': 1, 'Q': 2}

df['Embarked_num'] = df.Age
df_test['Embarked_num'] = df_test.Age

for i in town_to_number:
    df.loc[df.Embarked == i, 'Embarked_num'] = float(town_to_number[i])
    df_test.loc[df_test.Embarked == i, 'Embarked_num'] = float(town_to_number[i])

cleaned_df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)
cleaned_df_test = df_test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)

# Command below removes rows with empty observations
#cleaned_df = cleaned_df.dropna()

train_data = cleaned_df.values
test_data = cleaned_df_test.values

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[:,1:], train_data[:,0])

output = forest.predict(test_data)

passenger_id = df_test.values
passenger_id = passenger_id[:, 0]
output = [[passenger_id[x], output[x]] for x in range(len(passenger_id))]
output = np.array(output)
columns = ['PassengerId', 'Survived']

output_df = pd.DataFrame(data = output, columns = columns)
output_df.to_csv('randomforestmodel.csv', index = False, float_format = '%.f')

# Test accuracy on training data (just for fun)
train_output = forest.predict(train_data[:, 1:])

testing_accuracy = [1 - abs(train_output[x] - train_data[x,0]) for x in range(len(train_output))]

print(sum(testing_accuracy)/len(testing_accuracy))
