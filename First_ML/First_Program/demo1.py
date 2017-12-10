from sklearn import tree
from sklearn import svm

#height, weight, shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

#Data that prediction will run on
testdata = [192,92,45]

# Tree Classifier
clftree = tree.DecisionTreeClassifier()

clftree = clftree.fit(X,Y)

# SVM Classifier
clfsvm = svm.SVC()

clfsvm = clfsvm.fit(X,Y)

#Predictions of Different Classifiers
predictiontree = clftree.predict([testdata])

predictionsvm = clfsvm.predict([testdata])

#Print of best prediction
print('Tree Prediction')
print(predictiontree)

print('svm Prediction')
print(predictionsvm)
