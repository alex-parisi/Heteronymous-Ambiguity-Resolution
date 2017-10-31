
"""
Created on Sat Apr 22 14:10:06 2017
"""

import numpy as np
from nltk.util import ngrams
import nltk
import re
import pos
import os

testDir = os.path.dirname(os.path.realpath('__file__'))
testPath = os.path.join(testDir,'../data/OANC.txt')
testPath = os.path.abspath(os.path.realpath(testPath))
testData = open(testPath).read()


def runNgram(testData):
    #list of heteronyms
    heteronym = ['object', 'minute', 'conduct', 'close', 'use']
    
    #test data
    #testData = input('\nEnter test data:\n')
    
    #cleaning test data
    
    clean = re.compile('<.*?>')
    cleaned = re.sub(clean, '', testData)
    testData = re.sub("([^\w']|_)+",' ', cleaned).lower().split()
    
    #tagging parts of speech
    taggedText =  nltk.pos_tag(testData)
    
    
    ### Assign labels based on part of speech of heteronym ###
     
    # For each heteronym, loop through text and look for heteronym
    # if it exists, look for the predicted part of speech in the correspoinding list
    # assign labels by matching the predicted pos with the one in the list
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    label_5 = []
    
    hetIndex =[]
    for hetType in range(len(heteronym)):           
        for txtIndex in range(len(testData)):
            if(testData[txtIndex] == heteronym[hetType]):
                hetIndex.append(txtIndex)      
        if(len(hetIndex)!=0):
            if hetType==0:
                pos_het = pos.pos_1
            elif hetType ==1:
                pos_het = pos.pos_2
            elif hetType ==2:
                pos_het = pos.pos_3
            elif hetType ==3:
                pos_het = pos.pos_4
            elif hetType ==4:
                pos_het = pos.pos_5
            label = []   
            for i in range(len(hetIndex)):
                tg = taggedText[hetIndex[i]][1]
                for j in range(len(pos_het)):
                    ph = pos_het[j][0]
                    if (tg  == ph):
                        label.append((pos_het[j])[1])
            if hetType==0:
                label_1 = label
            elif hetType ==1:
                label_2 = label
            elif hetType ==2:
                label_3 = label
            elif hetType ==3:
                label_4 = label
            elif hetType ==4:
                label_5 = label                
        hetIndex = []
    
    labelAll = [label_1, label_2, label_3, label_4, label_5]
    #labelAll = filter(None, labelAll)
    return labelAll, taggedText
