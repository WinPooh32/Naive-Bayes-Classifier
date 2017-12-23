import math
import csv

def calcMean(propData, propIdx):
    sumProps = 0
    
    for x in propData:
        sumProps += x[propIdx]
    
    return sumProps/len(propData)


def calcVariance(propData, mean, propIdx):
    k = 1.0 / (len(propData) - 1)
    sumProps = 0
    
    for x in propData:
       sumProps += (x[propIdx] - mean)**2
    
    return k * sumProps


def calcGauss(classData):
    Gauss = []
    propsCount = len(classData[0])
    
    for propIdx in range(0, propsCount):
        mean = calcMean(classData, propIdx)
        variance = calcVariance(classData, mean, propIdx)
        
        Gauss.append( (mean, variance) )

    return Gauss


# Плонтость
def prob(propSample, propGauss):
    mean = propGauss[0]
    variance = propGauss[1]
    
    exp = math.exp(-((propSample - mean) ** 2) / (2 * variance))
    sqrt = 1.0/math.sqrt(2.0 * math.pi * variance)
    
    return exp * sqrt

################################################################################

def parseData(row):
    data = []
    dataLen = len(row)
    
    className = row[-1]
    
    for idx in range(0, dataLen - 1):
        val = 0
        
        try:
            val = float(row[idx])
        except ValueError:
            val = ord(row[idx]) / 10
            
        data.append(val)

    return className, data
    
    
def populateData(file):
    Data = {}
    
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        for row in reader:
            className, props = parseData(row)
            
            if Data.get(className) is None :
                Data[className] = []
            
            Data[className].append(props)
    
    return Data
    
    
def countAllRows(classesData):
    sum = 0
    
    for _, data in classesData.items():
        sum += len(data)
    return sum


def calcGaussians(classesData):
    Gaussians = {}
    
    for name, data in classesData.items():
        Gaussians[name] = calcGauss(data)
        
    return Gaussians
    
    
def calcClassProbability(classesData):
    ClassProb = {}
    
    allItems = countAllRows(classesData)

    for name, data in classesData.items():
        ClassProb[name] = len(data) / allItems

    return ClassProb
    

def learn(file):
    Data = populateData(file)
    return Data, calcGaussians(Data), calcClassProbability(Data)


def getMaxProbClass(probabilities):
    maxClass = ''
    
    for className, value in probabilities.items():
        if probabilities.get(maxClass) is None:
            maxClass = className
            
        elif probabilities[maxClass] < value:
            maxClass = className
            
    return maxClass


def predict(sample, Gaussians, ClassProb):
    Probs = {}
    
    for className, gauss in Gaussians.items():
        Probs[className] = ClassProb[className]
            
        propIdx = 0
        for sampleProp in sample:
            Probs[className] *= prob(sampleProp, gauss[propIdx])
            propIdx += 1
    
    return getMaxProbClass(Probs)


#==============================================================================#
Data, gauss, clProb = learn('Iris.csv') #'sobaki_i_volki.csv' #'Iris.csv'
TestData, _, _ =  learn('Iris_test.csv')

count = 0
successPredictions = 0

print("Реальный  |  Угаданный\n")

for className, rowData in TestData.items():
    for props in rowData:
        predicted = predict(props, gauss, clProb)
        if className == predicted:
            successPredictions += 1
        else:
            print(className + " != " + predicted)
        
        count += 1

print()
print("Точность классификации: " + "{0:.2f}".format((successPredictions/count) * 100) + "%")
