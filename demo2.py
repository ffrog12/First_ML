from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#height, weight, shoe size
Xtrain = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Ytrain = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

#Data that prediction will run on

testdata = [192,92,47]

# Tree Classifier

clftree = tree.DecisionTreeClassifier()
clftree = clftree.fit(Xtrain,Ytrain)

# SVM Classifier

clfsvm = svm.SVC()
clfsvm = clfsvm.fit(Xtrain,Ytrain)

#Gaussian Naive Bayesian Classifier

clfgnb = GaussianNB()
clfgnb = clfgnb.fit(Xtrain,Ytrain)

#Predictions of Different Classifiers

predictiontree = clftree.predict([testdata])
predictionsvm = clfsvm.predict([testdata])
predictiongnb = clfgnb.predict([testdata])

#Test which prediction is best
#validation data

Xtest = [[190, 70, 43], [150, 70, 30], [170, 80, 38], [200, 90, 44], [160, 50, 32]]
Ytest = ['male', 'female', 'male', 'male', 'female']

#Trialling against validation data

testpredtree = clftree.predict(Xtest)
testpredsvm = clfsvm.predict(Xtest)
testpredgnb = clfgnb.predict(Xtest)

#Scoring against validation

treeacc = accuracy_score(testpredtree, Ytest)
svmacc = accuracy_score(testpredsvm, Ytest)
gnbacc = accuracy_score(testpredgnb, Ytest)

#Print of best prediction

if treeacc > svmacc and treeacc > gnbacc:
          print('Tree Prediction {}, accuracy score {}%'.format(predictiontree,treeacc * 100))

elif svmacc > gnbacc and svmacc > treeacc:
          print('SVM Prediction {}, accuracy score {}%'.format(predictionsvm,svmacc * 100))

else :
          print('GNB Prediction{}, accuracy score {}%'.format(predictiongnb,gnbacc * 100))
