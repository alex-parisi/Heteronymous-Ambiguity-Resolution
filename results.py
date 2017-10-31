import os
from openpyxl import load_workbook
import numpy as np
import copy

import runTree
import bayesian
import ngram_tagger
import runEnsemble

testDir = os.path.dirname(os.path.realpath('__file__'))
testPath = os.path.join(testDir,'../data/OANC.txt')
testPath = os.path.abspath(os.path.realpath(testPath))
#testData = open(testPath).read()#.split()
testData = "I want you to tell me what to cook, close the door and spend the afternoon just hanging out wiht me"

# Get 4 prediction
treePred, indT = runTree.Run(testData)
ngramPred, indN = ngram_tagger.runNgram(testData)
bayesPred, indB = bayesian.Run(testData)
forestPred, indF = runEnsemble.Run(testData)

# Consistancy -- comment out if we use different testing methods
newB = []
for a in range(len(treePred)):
    newB.append(bayesPred[a*40:a*40+40])
    
bayesPred = newB

def crossValidation(res1, res2, type):
    percent = []
    if(type == 'accuracy'):
        correct, b = 0, 0
        for a in [0,1,3,4]:
            correct = len([i for i, j in zip(res1[a], res2[b]) if i == j])
            percent.append(correct/40.0*100.0)
            b += 1
    else:
        correct = 0
        for a in [0,1,3,4]:
            correct = len([i for i, j in zip(res1[a], res2[a]) if i == j])
            percent.append(correct/40.0*100.0)
    return percent

testDir = os.path.dirname(os.path.realpath('__file__'))
labPath = os.path.join(testDir, '../data/testLabels.xlsx')
wb = load_workbook(labPath)
ws = wb['Sheet1']

OANC_labels = []
for row in ws.rows:
    for cell in row:
        OANC_labels.append(cell.value)
        
hetTest = []
labelTest = []
tmpLabel = []
for i in range(len(OANC_labels)):
    if type(OANC_labels[i]) == unicode:
        hetTest.append(str(OANC_labels[i]).lower())
        if i != 0:
            labelTest.append(tmpLabel)
        tmpLabel = []
    else:
        if OANC_labels[i] is not None:
            tmpLabel.append(int(OANC_labels[i]))
        if i == (len(OANC_labels) - 1):
            labelTest.append(tmpLabel)        
del wb, ws, OANC_labels, tmpLabel    
    
# Get crossed validations
accuracy = []
accuracy.append(crossValidation(treePred,labelTest,'accuracy'))
accuracy.append(crossValidation(ngramPred,labelTest,'accuracy'))
accuracy.append(crossValidation(bayesPred,labelTest,'accuracy'))
accuracy.append(crossValidation(forestPred,labelTest,'accuracy'))

crossedVal = []
crossedVal.append(crossValidation(treePred,ngramPred,'not'))
crossedVal.append(crossValidation(treePred,bayesPred,'not'))
crossedVal.append(crossValidation(treePred,forestPred,'not'))
crossedVal.append(crossValidation(ngramPred,bayesPred,'not'))
crossedVal.append(crossValidation(ngramPred,forestPred,'not'))
crossedVal.append(crossValidation(forestPred,bayesPred,'not'))
 
#print()

#finalPrediction = copy.deepcopy(treePred)
#for i in range(len(treePred)):
#    for j in range(len(treePred[i])):
#        finalPrediction[i][j] = np.round((treePred[i][j]+bayesPred[i][j] + ngramPred[i][j])/3.0)
#accTot = crossValidation(finalPrediction,labelTest,'accuracy')


    
