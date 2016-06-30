import pandas as pd
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv', header=0)

df = df.drop(['Descript', 'Resolution'], axis = 1)

# Command below removes rows with empty observations
#cleaned_df = cleaned_df.dropna()
crime_categories = sorted(pd.Series.unique(df['Category']))
print(crime_categories)

lon_limits = [-122.5226, -122.3496]
lat_limits = [37.7007, 37.8152]

bins_lon = np.linspace(lon_limits[0], lon_limits[1], 10)
bins_lat = np.linspace(lat_limits[0], lat_limits[1], 10)

lon_boxes = [x for x in range(1, len(bins_lon))]
lat_boxes = [x for x in range(1, len(bins_lat))]

df['X_binned'] = pd.cut(df['X'], bins_lon, labels = lon_boxes)
df['Y_binned'] = pd.cut(df['Y'], bins_lat, labels = lat_boxes)

print(pd.value_counts(df['X_binned']))
print(pd.value_counts(df['Y_binned']))

df[['Address2', 'Address3']] = df['Address'].str.split('of|/', expand = True)
#df['Address2'] = df['Address2'].map(str.strip)
#df['Address3'] = df['Address3'].map(str.strip)
df['Address2'] = pd.core.strings.str_strip(df['Address2'])
df['Address3'] = pd.core.strings.str_strip(df['Address3'])

df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']] = df['Dates'].str.split('-| |:', expand = True).astype(int)

print(df.head(10))

#train_data = cleaned_df.values
#test_data = cleaned_df_test.values

#forest = RandomForestClassifier(n_estimators = 100)
#forest = forest.fit(train_data[:,1:], train_data[:,0])

#output = forest.predict(test_data)

#passenger_id = df_test.values
#passenger_id = passenger_id[:, 0]
#output = [[passenger_id[x], output[x]] for x in range(len(passenger_id))]
#output = np.array(output)
#columns = ['PassengerId', 'Survived']

#output_df = pd.DataFrame(data = output, columns = columns)
#output_df.to_csv('randomforestmodel.csv', index = False, float_format = '%.f')

# Test accuracy on training data (just for fun)
#train_output = forest.predict(train_data[:, 1:])

#testing_accuracy = [1 - abs(train_output[x] - train_data[x,0]) for x in range(len(train_output))]

#print(sum(testing_accuracy)/len(testing_accuracy))