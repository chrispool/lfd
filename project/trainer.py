import os, re
from xml.dom import minidom
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from transformers import *

'''Trains all classifiers and pickles the classifier for use in webpage'''



class Trainer:
	def __init__(self):
		#self.trainDutch()
		self.trainEnglish()
		#self.trainItalian()
		#self.trainSpanish()


	def readTrainingData(self, lang):
		X, Y = [], []
		if os.path.isdir("training/" + lang):
			indir = "training/" + lang
			
			#read truth file
			truthDict = {}
			with open(indir + '/truth.txt', encoding='utf-8') as f:
				for line in f:
					elements = line.strip().split(":::")
					truthDict[elements[0]] = {'sex' : elements[1], 'age' : elements[2], 'f1': elements[3], 'f2' : elements[4], 'f3': elements[5], 'f4' : elements[6], 'f5': elements[7]}
			#read XML files
			for root, dirs, filenames in os.walk(indir):
				for f in filenames:
					if f.endswith('xml'):
						x, y = self.parseXML(indir + '/' + f, truthDict)
						X.extend(x)
						Y.extend(y)

		
		
		#split in train/test
		split_point = int(0.75*len(X))
		trainX = X[:split_point]
		trainY = Y[:split_point]
		testX = X[split_point:]
		testY = Y[split_point:]

		return trainX, trainY, testX, testY
					

	def parseXML(self,doc,truthDict):
		xmldoc = minidom.parse(doc)
		xmlID = doc.split("/")[-1][:-4]
		
		itemlist = xmldoc.getElementsByTagName('document')
		x = []
		y = []
		for s in itemlist:
			x.append(self.processTweet(s.firstChild.nodeValue.strip()))
			y.append(truthDict[xmlID])
		return x,y


	def processTweet(self, tweet):
		#Convert to lower case
		tweet = tweet.lower()
		#Convert www.* or https?://* to URL
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
		#Convert @username to AT_USER
		tweet = re.sub('@[^\s]+','AT_USER',tweet)
		#Remove additional white spaces
		tweet = re.sub('[\s]+', ' ', tweet)
		#Replace #word with word
		tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
		#trim
		tweet = tweet.strip('\'"')
		return tweet


	def trainDutch(self):
		print("Dutch classifier")
		trainX,trainY,testX,testY = self.readTrainingData('dutch')
		self.trainSexClassifier(trainX,trainY,testX,testY)
		print()

	def trainEnglish(self):
		print("English classifier")
		trainX,trainY,testX,testY = self.readTrainingData('english')
		self.trainSexClassifier(trainX,trainY,testX,testY)
		self.trainAgeClassifier(trainX,trainY,testX,testY)
		print()
	
	def trainItalian(self):
		print("Italian classifier")
		trainX,trainY,testX,testY = self.readTrainingData('italian')
		self.trainSexClassifier(trainX,trainY,testX,testY)
		print()
	
	def trainSpanish(self):
		print("Spanish classifier")
		trainX,trainY,testX,testY = self.readTrainingData('spanish')
		self.trainSexClassifier(trainX,trainY,testX,testY)
		self.trainAgeClassifier(trainX,trainY,testX,testY)
		print()


	def trainSexClassifier(self,trainX,trainY,testX,testY):
		#get correct labels from dictionary in trainY and testY
		trainY = self.getYlabels(trainY, 'sex')
		testY = self.getYlabels(testY, 'sex')
		

		vec = TfidfVectorizer()
		pipeline = Pipeline([
			('features', FeatureUnion([
				('ngram_tf_idf', Pipeline([
					('counts', CountVectorizer()),
					('tf_idf', TfidfTransformer())
				])),
				('tweetstats', Pipeline([
					('stats', TextStats()),  # returns a list of dicts
                	('vect', DictVectorizer()),  # list of dicts -> feature matrix
				])),
				 # returns a list of dicts
                  # list of dicts -> feature matrix,
			])),
			('classifier', MultinomialNB())
		])


		#classifier = Pipeline( [('vec', vec), ('cls', MultinomialNB())] )
		#classifier = Pipeline( [('vec', vec), ('cls', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))] )
		pipeline.fit(trainX, trainY)
		predictY = pipeline.predict(testX)
		print(accuracy_score(testY, predictY))
		print(classification_report(testY, predictY))
	


	def trainAgeClassifier(self,trainX,trainY,testX,testY):
		#get correct labels from dictionary in trainY and testY
		trainY = self.getYlabels(trainY, 'age')
		testY = self.getYlabels(testY, 'age')
		
		vec = TfidfVectorizer()
		pipeline = Pipeline([
			('features', FeatureUnion([
				('ngram_tf_idf', Pipeline([
					('counts', CountVectorizer()),
					('tf_idf', TfidfTransformer())
				])),
				('tweetstats', Pipeline([
					('stats', TextStats()),  # returns a list of dicts
                	('vect', DictVectorizer()),  # list of dicts -> feature matrix
				])),
				 # returns a list of dicts
                  # list of dicts -> feature matrix,
			])),
			('classifier', MultinomialNB())
		])


		#classifier = Pipeline( [('vec', vec), ('cls', MultinomialNB())] )
		#classifier = Pipeline( [('vec', vec), ('cls', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))] )
		pipeline.fit(trainX, trainY)
		predictY = pipeline.predict(testX)
		print(accuracy_score(testY, predictY))
		print(classification_report(testY, predictY))


	def getYlabels(self, y, value):
		return [row[value] for row in y]
			
t = Trainer()
