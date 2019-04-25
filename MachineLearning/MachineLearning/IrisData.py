import pdb
import pandas as pd
import numpy as np
import random
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import itertools as it

def separateData(dataByClass, classNames):
	dataSetTraining = []
	dataSetTest = []
	##0 = Virginica, #1 = Setosa, #2 = Versicolor
	#seperate the data into training data and test data sets, and convert from strings to floats
	#Iris-virginica
	sepalLen0 = map(float, cleanLists(dataByClass[0][0]))
	sepalWid0 = map(float, cleanLists(dataByClass[0][1]))
	petalLen0 = map(float, cleanLists(dataByClass[0][2]))
	petalWid0 = map(float, cleanLists(dataByClass[0][3]))
	
	#Iris-Setosa
	sepalLen1 = map(float, cleanLists(dataByClass[1][0]))
	sepalWid1 = map(float, cleanLists(dataByClass[1][1]))
	petalLen1 = map(float, cleanLists(dataByClass[1][2]))
	petalWid1 = map(float, cleanLists(dataByClass[1][3]))
	
	#Iris-Versicolor
	sepalLen2 = map(float, cleanLists(dataByClass[2][0]))
	sepalWid2 = map(float, cleanLists(dataByClass[2][1]))
	petalLen2 = map(float, cleanLists(dataByClass[2][2]))
	petalWid2 = map(float, cleanLists(dataByClass[2][3]))
	
	#divide the original dataset, 2/3rds go to the training set, 1/3 saved for test set
	dataSize = len(sepalLen0)
	trainSize = math.ceil(dataSize * (2.0/3.0))
	trainSize = int(trainSize)
	
	#separate data into training and testing sets
	sepalLen0 = [sepalLen0[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalLen0) + trainSize - 1) // trainSize)]
	sepalLen1 = [sepalLen1[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalLen1) + trainSize - 1) // trainSize)]
	sepalLen2 = [sepalLen2[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalLen2) + trainSize - 1) // trainSize)]
	sepalLen0TrainSet = sepalLen0[0]
	sepalLen1TrainSet = sepalLen1[0]
	sepalLen2TrainSet = sepalLen2[0]
	sepalLen0TestSet = sepalLen0[1]
	sepalLen1TestSet = sepalLen1[1]
	sepalLen2TestSet = sepalLen2[1]
	
	sepalWid0= [sepalWid0[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalWid0) + trainSize - 1) // trainSize)]
	sepalWid1= [sepalWid1[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalWid1) + trainSize - 1) // trainSize)]
	sepalWid2= [sepalWid2[i *trainSize:(i + 1) * trainSize] for i in range((len(sepalWid2) + trainSize - 1) // trainSize)]
	sepalWid0TrainSet = sepalWid0[0]
	sepalWid1TrainSet = sepalWid1[0]
	sepalWid2TrainSet = sepalWid1[0]
	sepalWid0TestSet = sepalWid0[1]
	sepalWid1TestSet = sepalWid1[1]
	sepalWid2TestSet = sepalWid1[1]
	
	petalLen0= [petalLen0[i *trainSize:(i + 1) * trainSize] for i in range((len(petalLen0) + trainSize - 1) // trainSize)]
	petalLen1= [petalLen1[i *trainSize:(i + 1) * trainSize] for i in range((len(petalLen1) + trainSize - 1) // trainSize)]
	petalLen2= [petalLen2[i *trainSize:(i + 1) * trainSize] for i in range((len(petalLen2) + trainSize - 1) // trainSize)]
	petalLen0TrainSet = petalLen0[0]
	petalLen1TrainSet = petalLen1[0]
	petalLen2TrainSet = petalLen2[0]
	petalLen0TestSet = petalLen0[1]
	petalLen1TestSet = petalLen1[1]
	petalLen2TestSet = petalLen2[1]
	
	petalWid0 = [petalWid0[i *trainSize:(i + 1) * trainSize] for i in range((len(petalWid0) + trainSize - 1) // trainSize)]
	petalWid1 = [petalWid1[i *trainSize:(i + 1) * trainSize] for i in range((len(petalWid1) + trainSize - 1) // trainSize)]
	petalWid2 = [petalWid2[i *trainSize:(i + 1) * trainSize] for i in range((len(petalWid2) + trainSize - 1) // trainSize)]
	petalWid0TrainSet = petalWid0[0]
	petalWid1TrainSet = petalWid1[0]
	petalWid2TrainSet = petalWid2[0]
	petalWid0TestSet = petalWid0[1]
	petalWid1TestSet = petalWid1[1]
	petalWid2TestSet = petalWid2[1]
	
	#now store data by classification and return
	train0Sepal = [sepalLen0TrainSet, sepalWid0TrainSet]
	train0Petal = [petalLen0TrainSet, petalWid0TrainSet]
	
	train1Sepal = [sepalLen1TrainSet, sepalWid1TrainSet]
	train1Petal = [petalLen1TrainSet, petalWid1TrainSet]
	
	train2Sepal = [sepalLen2TrainSet, sepalWid2TrainSet]
	train2Petal = [petalLen2, petalWid2TrainSet]
	
	test0Sepal = [sepalLen0TestSet, sepalWid0TestSet]
	test0Petal = [petalLen0TestSet, petalWid0TestSet]
	
	test1Sepal = [sepalLen1TestSet, sepalWid1TestSet]
	test1Petal = [petalLen1TestSet, petalWid1TestSet]
	
	test2Sepal = [sepalLen2TestSet, sepalWid2TestSet]
	test2Petal = [petalLen2TestSet, petalWid2TestSet]
	
	trainData = [train0Sepal, train0Petal, train1Sepal, train1Petal, train2Sepal, train2Petal]
	testData = [test0Sepal, test0Petal, test1Sepal, test1Petal, test2Sepal, test2Petal]

	return trainData, testData

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

def linearRegressionManager(trainSet, testSet, alpha = 0.0001):
	count = 0 # 0 = Virginica, 1 = Setosa, 2 = Versicolor
	for i,j in it.izip(trainSet, testSet):	
		x_train = i[0]
		y_train = i[1]
		x_test = j[0]
		y_test = j[1]
		
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test = np.array(x_test)
		y_test = np.array(y_test)
		
		x_train = x_train.reshape(-1,1)
		x_test = x_test.reshape(-1,1)
		n = len(x_train)
		linearRegressionCalc(x_train, y_train, n, alpha)

def linearRegressionCalc(x_train, y_train, n, alpha):
	print x_train
	a_0 = np.zeros((n, 1))
	a_1 = np.zeros((n, 1))
	
	epochs = 0
	while(epochs < 1000):
		y = a_0 +a_1 * x_train
		error = y - y_train
		mean_sq_er = np.sum(error**2)
		mean_sq_er = mean_sq_er/n
		a_0 = a_0 - alpha * 2 * np.sum(error)/n
		a_1 = a_1 - alpha * 2 * np.sum(error * x_train)/n
		epochs += 1
		if epochs%10 == 0:
			print mean_sq_er
	pdb.set_trace()
	print "a_0, a_1"
	print a_0
	print a_1
		
	
def getXY(set):
	x = []
	y = []
	for i in set:
		if len(i) == 2:
			x1 = i[0]
			y1= i[1]
			
			x1 = np.asarray(x1)
			y1 = np.asarray(y1)
			
			x.append(x1.reshape(-1,1))
			y.append(y1)
	return(x, y)

def splitClass(file, classes, delim = ','):
	dataByClass = []
	term = []
	i = 0
	
	for i in classes:
		classStore = [[],[],[],[]]
		dataByClass.append(classStore)
	
	with open(file) as f:
		for line in f:
			j = 0
			for i in classes:
				if i in line:
					cells = line.split(delim)
					term0 = cells[0]
					term1 = cells[1]
					term2 = cells[2]
					term3 = cells[3]
					dataByClass[j][0].append(term0)
					dataByClass[j][1].append(term1)
					dataByClass[j][2].append(term2)
					dataByClass[j][3].append(term3)
					break
				j = j+1
	return dataByClass
				
def main():
	irisData = "/mnt/c/Users/kdatin01/Desktop/MachineLearning/datasets/iris.data"
	classNames = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
	dataByClass = splitClass(irisData, classNames)
	#randomFile = randomizeText(irisData)
	#irisData = open("/mnt/c/Users/kdatin01/Desktop/MachineLearning/datasets/iris.data", 'r')
	
	
	trainSet, testSet = separateData(dataByClass, classNames)
	linearRegressionManager(trainSet, testSet)
	irisData.close()
	

if __name__ == "__main__":
	main()
