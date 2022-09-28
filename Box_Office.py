import csv
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

#Code to create a new csv file with the features I want
"""
f = open('Netflix Preprocessed2.csv', 'w', newline='')
dataset = 'Netflix Dataset Latest 2021.csv'

csvreader = csv.reader(open(dataset))

header = next(csvreader)
print(header)
myRow = [header[17]] + header[1:17] + header[18:21] + [header[24]]
writer = csv.writer(f)
writer.writerow(myRow)

for line in csvreader:
    writer.writerow([line[17]] + line[1:17] + line[18:21] + [line[24]])

"""

dataset = 'Netflix Preprocessed2.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)

#x is an array of all the features used to predict Box Office
#y is an array of the Box Office
x = []
Y = []
for line in csvreader:
    if line[0] != '':
        x.append(line[1:])
        firstIntFound = False
        s = ''
        for c in line[0]:
            if c.isdigit():
                if firstIntFound:
                    s+='0'
                else:
                    s+= c
                    firstIntFound = True
        s = int(s)
        #separate into 3 different ranges of money
        #0 represents a box office value less than $10,000,000
        #1 represents a box office value between $10,000,000 - $50,000,000
        #2 represents a box office value greater than $50,000,000
        if s < 10000000:
            Y.append(0)
        elif s < 50000000:
            Y.append(1)
        else:
            Y.append(2)

print(Y.count(0), Y.count(1), Y.count(2))

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
