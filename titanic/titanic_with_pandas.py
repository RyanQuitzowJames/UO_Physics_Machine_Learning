import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv('train.csv', header=0)

print(df.head(3))
print(df.tail(3))

print(df.dtypes)

print(df.info())
print(df.describe())

print(df['Age'][0:10])
print(df.Age[0:10])
print(df.Age[:10])

print(df[['Sex', 'Pclass', 'Age']][:10])

print(df[df.Age > 60])
print(df[df.Age.isnull()][['Sex', 'Pclass', 'Age']][:10])

print([[i, len(df[(df.Sex == 'male') & (df.Pclass == i)])] for i in range(1,4)])

df.Age.hist()
pl.show()

df.Age.dropna().hist(bins=16, range=(0,80), alpha = 0.5)
pl.show()

df['Gender'] = df.Sex.map({'female': 0, 'male': 1}).astype(int)

median_ages = [
    [
        df[(df.Gender == i) & (df.Pclass == j+1)].Age.dropna().median()
    for j in range(3)]
for i in range(2)]

median_ages = np.array(median_ages)
print(median_ages)

df['AgeFill'] = df.Age

for i in range(2):
    for j in range(3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10))

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

df.AgeFill.dropna().hist(bins=16, range=(0,80), alpha = 0.5)
pl.show()
df.AgeIsNull.dropna().hist(alpha = 0.5)
pl.show()
df.FamilySize.dropna().hist(alpha = 0.5)
pl.show()
df['Age*Class'].dropna().hist(bins=16, alpha = 0.5)
pl.show()

cleaned_df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# Command below removes rows with empty observations
#cleaned_df = cleaned_df.dropna()

print(df.dtypes)
print(cleaned_df.dtypes)

train_data = cleaned_df.values
