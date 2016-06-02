import csv
import numpy as np

training_data_path = 'train.csv'
test_data_path = 'test.csv'

with open(training_data_path) as infile:
    read_csv = csv.reader(infile)
    raw_data = [x for x in read_csv]
    header = raw_data[0]
    data = raw_data[1:]
    data = np.array(data)

women_only_stats = data[:,4] == 'female' # Rows with women
men_only_stats = data[:,4] == 'male' # Row with men

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived = np.sum(women_onboard)/np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)

print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)

with open(test_data_path) as infile:
    readCSV = csv.reader(infile)
    raw_data = [x for x in readCSV]
    header = raw_data[0]
    data = raw_data[1:]
    data = np.array(data)

prediction = [['PassengerId', 'Survived']] + [[row[0], '1'] if row[3] == 'female' else [row[0], '0'] for row in data]

prediction_file_path = 'genderbasedmodel.csv'
with open(prediction_file_path, 'w') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerows(prediction)
