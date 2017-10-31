## Collecting .sgm files into one .txt file
import re
fileName =  'reuter_data/reut2-0'
fileNum = ["%.2d" % i for i in range(22)]
textfile = open('textDataAll.txt', 'w')
for i in fileNum:
    filePath = fileName+'%s.sgm'%i   
    f = open(filePath, 'r')
    data= f.read()
    noSGMtags = re.compile('<.*?>')
    clean = re.sub(noSGMtags, '', data)
    text = re.sub("([^\w']|_)+",' ',clean).lower().split()
    textfile.writelines("%s " % item for item in text)
    f.close()
textfile.close()


## Import excel data of labels
from openpyxl import load_workbook
import numpy as np

wb = load_workbook(filename = 'Labels_test.xlsx')
ws = wb['Sheet1']

labels = []
for row in ws.rows:
    for cell in row:
        labels.append(cell.value)

numWords = 6
heteronyms = []
labelAll = []
for i in range(len(labels)):
    if type(labels[i]) == unicode:
        heteronyms.append(labels[i])
        if i != 0:
            labelAll.append(tmpLabel)
        tmpLabel = []
    else:
        if labels[i] is not None:
            tmpLabel.append(int(labels[i]))
        if i == (len(labels) - 1):
            labelAll.append(tmpLabel)


## Testing data: GloWbe
fileName = 'dataset.txt'
textfile = open('testDataNL.txt','w')
f = open(fileName,'r')
data = f.readlines()

testData = []
for line in data:
    noTags = re.compile('<.*?>')
    clean = re.sub(noTags,'',line)
    text = re.sub("([^\w']|_)+",' ',clean).lower().split()
    textTmp = text[1:-1]
    testData = testData + textTmp
del line, data, clean, text, textTmp
testData = '\n'.join(map(str,testData))
textfile.writelines("%s" % item for item in testData)
textfile.close()
f.close()


## Testing data: OANC Verbatim & Slate
import re
fileLoc =  'OANC_data/Verbatim/VOL'
fileNum1 = ["%.2d" % i for i in (range(15,20)+range(21,24))]
fileNum2 = ["%.1d" % i for i in range(1,5)]

OANC = open('OANC_NL.txt', 'w')
for i in fileNum1:
    for j in fileNum2:
        filePath = fileLoc+'%s_%s.txt'%(i,j)   
        f = open(filePath, 'r')
        data= f.read()
        noSGMtags = re.compile('<.*?>')
        clean = re.sub(noSGMtags, '', data)
        text = re.sub("([^\w']|_)+",' ',clean).lower().split()
        OANC.writelines("%s\n" % item for item in text)
        f.close()


## Testing data: OANC Slate
import re, os
fileLoc =  'OANC_data/Slate'

for files in os.listdir(fileLoc): 
    filePath = fileLoc+"/"+files
    f = open(filePath, 'r')
    data= f.read()
    noSGMtags = re.compile('<.*?>')
    clean = re.sub(noSGMtags, '', data)
    text = re.sub("([^\w']|_)+",' ',clean).lower().split()
    OANC.writelines("%s\n" % item for item in text)
    f.close()
OANC.close()


