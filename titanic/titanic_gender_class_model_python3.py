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

fare_ceiling = 39
data[data[:,9].astype(np.float) > fare_ceiling, 9] = fare_ceiling

fare_bracket_size = 10
number_of_price_brackets = np.ceil(fare_ceiling/fare_bracket_size)

number_of_classes = len(np.unique(data[:,2]))
number_of_genders = len(np.unique(data[:,4]))

gender_list = ['female', 'male']

survival_table = [
    [
        [np.mean(data[(data[:, 4] == gender_type)
            & (data[:, 2].astype(np.float) == class_num+1)
            & (price_bin*fare_bracket_size <= data[:,9].astype(np.float))
            & (data[:,9].astype(np.float) < (price_bin+1)*fare_bracket_size), 1].astype(np.float))
        for price_bin in range(int(number_of_price_brackets))]
    for class_num in range(int(number_of_classes))]
for gender_type in gender_list]

survival_table = np.array(survival_table)
survival_table[survival_table != survival_table] = 0

print(survival_table)

survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

print(survival_table)

with open(test_data_path) as infile:
    readCSV = csv.reader(infile)
    raw_data = [x for x in readCSV]
    header = raw_data[0]
    data = raw_data[1:]
    data = np.array(data)

def predict_survival(temp_row, table):
    if temp_row[8]:
        fare_bin = np.floor(float(temp_row[8])/fare_bracket_size)
        if fare_bin*fare_bracket_size > fare_ceiling:
            fare_bin = np.floor(fare_ceiling/fare_bracket_size)
    else:
        fare_bin = 3 - float(temp_row[1])
    gender_index = gender_list.index(temp_row[3])
    return int(table[gender_index, int(temp_row[1])-1, int(fare_bin)])

prediction = [['PassengerId', 'Survived']] + [[row[0], predict_survival(row, survival_table)] for row in data]

prediction_file_path = 'genderclassmodel.csv'
with open(prediction_file_path, 'w') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerows(prediction)
