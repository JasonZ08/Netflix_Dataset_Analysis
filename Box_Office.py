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

#Code to create a new csv file with the features I want

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