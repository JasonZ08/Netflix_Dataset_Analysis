import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
"""
#Code to create a new csv file with the features I want

f = open('Netflix Preprocessed.csv', 'w', newline='')
dataset = 'Netflix Dataset Latest 2021.csv'

csvreader = csv.reader(open(dataset))

header = next(csvreader)
print(header)
myRow = [header[5]] + header[1:5] + header[6:21] + [header[24]]
writer = csv.writer(f)
writer.writerow(myRow)

for line in csvreader:
    writer.writerow([line[5]] + line[1:5] + line[6:21] + [line[24]])
"""

#read in the dataset
dataset = 'Netflix Preprocessed.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)


#x is an array of all the features used to predict hidden gem score
#y is an array of the hidden gem score
x = []
Y = []
for line in csvreader:
    x.append(line[1:])
    if line[0] == '':
        Y.append(0)
    else:
        Y.append(int(float(line[0])))

#change categorical data into discrete data
x_Train, x_Test, y_Train, y_Test = train_test_split(x, Y, test_size=.2, train_size=.8, random_state=20)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(x_Train)
x_Train_enc = oe.transform(x_Train)
x_Test_enc = oe.transform(x_Test)

myDict = {}
for i in range(0, len(x_Train_enc[0])):
    myDict[i] = header[i + 1]
print(myDict)


#perform chi square test
chiSquareTest = SelectKBest(score_func=chi2, k="all")
chiSquareTest.fit(x_Train_enc, y_Train)
x_Train_fs = chiSquareTest.transform(x_Train_enc)
x_Test_fs = chiSquareTest.transform(x_Test_enc)
for i in range(len(chiSquareTest.scores_)):
    print('Feature %d: %f' % (i, chiSquareTest.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(chiSquareTest.scores_))], chiSquareTest.scores_)
pyplot.xlabel('Feature Number')
pyplot.ylabel('Chi-Squared Feature Importance')
pyplot.show()
#features 15, 18, 7, 17 and 19 were the top 5 most correlated feature to the hidden gem score

mutInfoTest = SelectKBest(score_func=mutual_info_classif, k="all")
mutInfoTest.fit(x_Train_enc, y_Train)
x_Train_fs = mutInfoTest.transform(x_Train_enc)
x_Test_fs = mutInfoTest.transform(x_Test_enc)
for i in range(len(mutInfoTest.scores_)):
    print('Feature %d: %f' % (i, mutInfoTest.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(mutInfoTest.scores_))], mutInfoTest.scores_)
pyplot.xlabel('Feature Number')
pyplot.ylabel('Mutual Information Feature Importance')
pyplot.show()
#features 11, 12, 9, 10, 1

featToRank = {}
featToName = {}
# create random forest classifier and determine attribute imporantance with RFE
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(x_Train_enc, y_Train)
for i in range(1, len(header)):
    featToRank[i-1] = selector.ranking_[i - 1]
    featToRank[i-1] = header[i]
    print(header[i], selector.ranking_[i-1])

print("   ", end = '')
for i in range(1, 10):
    print("Feature " + str(i-1) + ": " + str(selector.ranking_[i-1]) + ' | ', end = '')
print("Feature " + str(9) + ": " + str(selector.ranking_[9]))
print()
for i in range(11, len(header) - 1):
    print("Feature " + str(i-1) + ": " + str(selector.ranking_[i - 1]) + ' | ', end='')
print("Feature " + str(19) + ": " + str(selector.ranking_[19]))

#features 19, 11, 12 10, 0

"""
#create 3 new csv files, each one containing the top 5 features for the tests
f = open('Chi Squared Features.csv', 'w', newline='')
myRow = [header[0]] + [header[16]] + [header[19]] + [header[8]] + [header[18]] + [header[20]]
writer = csv.writer(f)
writer.writerow(myRow)
for i in range(len(x_Train_enc)):
    writer.writerow([y_Train[i]] + [x_Train_enc[i][15]] + [x_Train_enc[i][18]] + [x_Train_enc[i][7]] + [x_Train_enc[i][17]] + [x_Train_enc[i][19]])

for i in range(len(x_Test_enc)):
    writer.writerow(
        [y_Test[i]] + [x_Test_enc[i][15]] + [x_Test_enc[i][18]] + [x_Test_enc[i][7]] + [x_Test_enc[i][17]] + [
            x_Test_enc[i][19]])

f = open('Mutual Info Features.csv', 'w', newline='')
myRow = [header[0]] + [header[12]] + [header[13]] + [header[10]] + [header[11]] + [header[2]]
writer = csv.writer(f)
writer.writerow(myRow)
for i in range(len(x_Train_enc)):
    writer.writerow(
        [y_Train[i]] + [x_Train_enc[i][11]] + [x_Train_enc[i][12]] + [x_Train_enc[i][9]] + [x_Train_enc[i][10]] + [
            x_Train_enc[i][1]])

for i in range(len(x_Test_enc)):
    writer.writerow(
        [y_Test[i]] + [x_Test_enc[i][11]] + [x_Test_enc[i][12]] + [x_Test_enc[i][9]] + [x_Test_enc[i][10]] + [
            x_Test_enc[i][1]])

f = open('RFE Features.csv', 'w', newline='')
myRow = [header[0]] + [header[20]] + [header[12]] + [header[13]] + [header[11]] + [header[1]]
writer = csv.writer(f)
writer.writerow(myRow)
for i in range(len(x_Train_enc)):
    writer.writerow(
        [y_Train[i]] + [x_Train_enc[i][19]] + [x_Train_enc[i][11]] + [x_Train_enc[i][12]] + [x_Train_enc[i][10]] + [
            x_Train_enc[i][0]])

for i in range(len(x_Test_enc)):
    writer.writerow(
        [y_Test[i]] + [x_Test_enc[i][19]] + [x_Test_enc[i][11]] + [x_Test_enc[i][12]] + [x_Test_enc[i][10]] + [
            x_Test_enc[i][0]])
"""