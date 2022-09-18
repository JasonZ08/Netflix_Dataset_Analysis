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

x_Train, x_Test, y_Train, y_Test = train_test_split(x, Y, test_size=.2, train_size=.8, random_state=20)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(x_Train)
x_Train_enc = oe.transform(x_Train)
x_Test_enc = oe.transform(x_Test)

#perform chi square test
chiSquareTest = SelectKBest(score_func=chi2, k="all")
chiSquareTest.fit(x_Train_enc, y_Train)
x_Train_fs = chiSquareTest.transform(x_Train_enc)
x_Test_fs = chiSquareTest.transform(x_Test_enc)
for i in range(len(chiSquareTest.scores_)):
    print('Feature %d: %f' % (i, chiSquareTest.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(chiSquareTest.scores_))], chiSquareTest.scores_)
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
pyplot.show()
#features 11, 12, 9, 10, 1

featToRank = {}
# create random forest classifier and determine attribute imporantance with RFE
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(x_Train_enc, y_Train)
for i in range(1, len(header)):
    featToRank[header[i]] = selector.ranking_[i - 1]
    print(header[i], selector.ranking_[i-1])
print(featToRank)
#features 19, 11, 12 10, 0
