import pdb
import pandas as pd
import numpy as np
import random
import sys

def separateData(inFile, classNames):
	#seperate the data into training data and test data sets
	cols, index = getColumns(inFile)
	sepalLen = cols[index[0]]
	sepalWid = cols[index[1]]
	petalLen = cols[index[2]]
	petalWid = cols[index[3]]
	classif = cols[index[4]]
	
	#remove any empty strings from lists
	sepalLen = cleanLists(sepalLen)
	sepalWid = cleanLists(sepalWid)
	petalLen = cleanLists(petalLen)
	petalWid = cleanLists(petalWid)
	classif = cleanLists(classif)
	
	#convert sepal length and width, and petal legnth and width from strings to floats
	sepalLen = map(float, sepalLen)
	sepalWid = map(float, sepalWid)
	petalLen = map(float, petalLen)
	petalWid = map(float, petalWid)
	
	#divide the original dataset, 2/3rds go to the training set, 1/3 saved for test set
	dataSize = len(sepalLen)
	trainSize = dataSize * (2.0/3.0)
	trainSize = int(trainSize)
	
	#separate data into training and testing sets
	sepalLen = [sepalLen[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalLen) + trainSize - 1) // trainSize)]
	sepalLenTrainSet = sepalLen[0]
	sepalLenTestSet = sepalLen[1]
	
	sepalWid= [sepalWid[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalWid) + trainSize - 1) // trainSize)]
	sepalWidTrainSet = sepalWid[0]
	sepalWidTestSet = sepalWid[1]
	
	petalLen= [petalLen[i *trainSize:(i + 1) * trainSize] for i in range((len(petalLen) + trainSize - 1) // trainSize)]
	petalLenTrainSet = petalLen[0]
	petalLenTestSet = petalLen[1]
	
	petalWid = [petalWid[i *trainSize:(i + 1) * trainSize] for i in range((len(petalWid) + trainSize - 1) // trainSize)]
	petalWidTrainSet = petalWid[0]
	petalWidTestSet = petalWid[1]
	
	classif = [classif[i *trainSize:(i + 1) * trainSize] for i in range((len(classif) + trainSize - 1) // trainSize)]
	classifTrainSet = classif[0]
	classifTestSet = classif[1]
	
	xySepalTrain = [sepalLenTrainSet, sepalWidTrainSet]
	xyPetalTrain = [petalLenTrainSet, petalWidTrainSet]
	xySepalTest = [sepalLenTestSet, sepalWidTestSet]
	xyPetalTest = [petalLenTestSet, petalWidTestSet]
	
	dataSetTraining = [xySepalTrain, xyPetalTrain, classifTrainSet]
	dataSetTest = [xySepalTest, xyPetalTest, classifTestSet]

	return dataSetTraining, dataSetTest

def cleanLists(list):
	newList = filter(None, list)
	return newList
	
def getColumns(inFile, delim=',', header=False):
	#get columns of data from inFile. Order of rows is respected.
	cols = {}
	indexToName = {}
	for lineNum, line in enumerate(inFile):
		if lineNum == 0:
			headings = line.split(delim)
			i = 0
			for heading in headings:
				heading = heading.strip()
				if header:
					cols[heading] = []
					indexToName[i] = heading
				else:
					#heading is just a cell
					cols[i] = [heading]
					indexToName[i] = i	
				i += 1
		else:
			cells = line.split(delim)
			i = 0
			for cell in cells:
				cell = cell.strip()
				cols[indexToName[i]] += [cell]
				i += 1
	return cols, indexToName

def randomizeText(file):
	fileOrig = open(file, "r")
	randomLines = fileOrig.readlines()
	fileOrig.close()
	
	random.shuffle(randomLines)
	fileNew = open(file, "w")
	fileNew.writelines(randomLines)
	fileNew.close()

def linearRegressionManager(trainSet, testSet, n = 700, alpha = 0.0001):
	print len(trainSet)
	for i in trainSet:
		pdb.set_trace()
		if len(i) == 2:
			x_train = i[0]
			y_train = i[1]
			
			x_train = np.asarray(x_train)
			y_train = np.asarray(y_train)
			
			x_train = x_train.reshape(-1,1)
			
		
def main():
	irisData = "/mnt/c/Users/kdatin01/Desktop/MachineLearning/datasets/iris.data"
	randomFile = randomizeText(irisData)
	irisData = file("/mnt/c/Users/kdatin01/Desktop/MachineLearning/datasets/iris.data", 'r')
	classNames = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
	trainSet, testSet = separateData(irisData, classNames)
	linearRegressionManager(trainSet, testSet)
	irisData.close()
	

if __name__ == "__main__":
	main()
