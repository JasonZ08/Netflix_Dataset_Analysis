import csv
import random
import numpy as np
import pandas as pd

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
        Y.append(int(s))
print(len(x))
print(len(Y))
print(Y[0])
