from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = 'Chi Squared Features.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)

x = []
Y = []
for line in csvreader:
    x.append(line[1:])
    if line[0] == '':
        Y.append(0)
    else:
        Y.append(int(float(line[0])))

x_Train, x_Test, y_Train, y_Test = train_test_split(x, Y, test_size=.2, train_size=.8, random_state=20)
clf = RandomForestClassifier()
clf.fit(x_Train, y_Train)
predictValues = clf.predict(x_Test)
count = 0
print()
print('Chi-squared Test')
print(classification_report(y_Test, predictValues))

dataset = 'Mutual Info Features.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)

x = []
Y = []
for line in csvreader:
    x.append(line[1:])
    if line[0] == '':
        Y.append(0)
    else:
        Y.append(int(float(line[0])))

x_Train, x_Test, y_Train, y_Test = train_test_split(x, Y, test_size=.2, train_size=.8, random_state=20)
clf = RandomForestClassifier()
clf.fit(x_Train, y_Train)
predictValues = clf.predict(x_Test)
count = 0
print('Mutual Information')
print(classification_report(y_Test, predictValues))

dataset = 'RFE Features.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)

x = []
Y = []
for line in csvreader:
    x.append(line[1:])
    if line[0] == '':
        Y.append(0)
    else:
        Y.append(int(float(line[0])))

x_Train, x_Test, y_Train, y_Test = train_test_split(x, Y, test_size=.2, train_size=.8, random_state=20)
clf = RandomForestClassifier()
clf.fit(x_Train, y_Train)
predictValues = clf.predict(x_Test)
count = 0
print('Recursive Feature Elimination')
print(classification_report(y_Test, predictValues))