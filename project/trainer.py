import os, re, sys
from xml.dom import minidom
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter,defaultdict
from transformers import *




class Trainer:
	def __init__(self, argv):
		#seperate train and testset
		if len(argv) == 3:
			#read trainingData, list with 3 values, X, Y and DOCID
			self.englishTrainData = self.readData(argv[1],'english')
			self.dutchTrainData = self.readData(argv[1],'dutch')
			self.italianTrainData = self.readData(argv[1],'italian')
			self.spanishTrainData = self.readData(argv[1],'spanish')
			self.allTrainData = list(zip(self.englishTrainData, self.dutchTrainData, self.italianTrainData, self.spanishTrainData))
			
			#read all test data
			self.englishTestData = self.readData(argv[2],'english')
			self.dutchTestData = self.readData(argv[2],'dutch')
			self.italianTestData = self.readData(argv[2],'italian')
			self.spanishTestData = self.readData(argv[2],'spanish')
			self.allTestData = list(zip(self.englishTestData, self.dutchTestData, self.italianTestData, self.spanishTestData))

			
		
		#only trainset, split the data in train and test
		elif len(argv) == 2:
			enData = self.readData(argv[1],'english')
			nlData = self.readData(argv[1],'dutch')
			itData = self.readData(argv[1],'italian')
			spData = self.readData(argv[1],'spanish')
			allData = list(zip(self.englishTrainData, self.dutchTrainData, self.italianTrainData, self.spanishTrainData))
		
			splitpointEN = int(len(enData[0]) * 0.75)
			splitpointNL = int(len(nlData[0]) * 0.75)
			splitpointIT = int(len(itData[0]) * 0.75)
			splitpointSP = int(len(spData[0]) * 0.75)
			splitpointALL = int(len(allData[0]) * 0.75)
			
			self.englishTrainData = enData[:splitpointEN]
			self.dutchTrainData = nlData[:splitpointNL]
			self.italianTrainData = itData[:splitpointIT]
			self.spanishTrainData = spData[:splitpointSP]
			self.allTrainData = allData[:splitpointALL]

			self.englishTestData = enData[splitpointEN:]
			self.dutchTestData = nlData[splitpointNL:]
			self.italianTestData = itData[splitpointIT:]
			self.spanishTestData = spData[splitpointSP:]
			self.allTestData = allData[splitpointALL:]
		else:
			exit("usage trainer.py <trainset> *<testset>")

		#train and predict sex for each user, output is used for age classifier		
		self.sexClassifier()
		
		#train and predict age for each user, one of the features is the sex classifier			
		self.ageClassifier()
	
	def sexClassifier(self):
		'''general function that coordinates the sex classification'''
		#train classsifier on all languages using only style features to be used as feature
		sexClassifier = self.generalSexClassifier()
		
		#train models for each language using the general sex classifier as feature		
		self.evaluation(self.trainEnglishSexClassifier(sexClassifier), 'english')
		self.evaluation(self.trainDutchSexClassifier(sexClassifier), 'dutch')
		self.evaluation(self.trainItalianSexClassifier(sexClassifier), 'italian')
		self.evaluation(self.trainSpanishSexClassifier(sexClassifier), 'spanish')


	def generalSexClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.allTrainData[0][0]
		trainY = self.getYlabels(self.allTrainData[1][0], 'sex')
		testX = self.allTrainData[0][0]
		testY = self.getYlabels(self.allTestData[1][0], 'sex')
		

		combined_features = FeatureUnion([("caps", CountCaps()),				
										("wordCaps", CountWordCaps())])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='linear')
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		predictY = pipeline.predict(testX)

		return pipeline


	def trainEnglishSexClassifier(self, generalSexClassifier):
		#get correct labels from dictionary in trainY and testY
		trainX = self.englishTrainData[0]
		trainY = self.getYlabels(self.englishTrainData[1], 'sex')
		testX = self.englishTrainData[0]
		testY = self.getYlabels(self.englishTrainData[1], 'sex')
		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(max_features=1000)),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char", min_df=4, max_features=1000)), 
										("counts", CountVectorizer(max_features=1000)),
										("generalSexClassifier", Classifier(generalSexClassifier)),
										])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='linear')
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		predictY = pipeline.predict(testX)
		return predictY

	
	def trainDutchSexClassifier(self, generalSexClassifier):
		#get correct labels from dictionary in trainY and testY
		trainX = self.dutchTrainData[0]
		trainY = self.getYlabels(self.dutchTrainData[1], 'sex')
		testX = self.dutchTrainData[0]
		testY = self.getYlabels(self.dutchTrainData[1], 'sex')
		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(max_features=1000)),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char", min_df=4, max_features=1000)), 
										("counts", CountVectorizer(max_features=1000)),
										("generalSexClassifier", Classifier(generalSexClassifier)),
										])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='linear')
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		predictY = pipeline.predict(testX)
		return predictY


	def trainItalianSexClassifier(self, generalSexClassifier):
		#get correct labels from dictionary in trainY and testY
		trainX = self.italianTrainData[0]
		trainY = self.getYlabels(self.italianTrainData[1], 'sex')
		testX = self.italianTrainData[0]
		testY = self.getYlabels(self.italianTrainData[1], 'sex')
		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(max_features=1000)),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char", min_df=4, max_features=1000)), 
										("counts", CountVectorizer(max_features=1000)),
										("generalSexClassifier", Classifier(generalSexClassifier)),
										])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='linear')
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		predictY = pipeline.predict(testX)
		return predictY

	def trainSpanishSexClassifier(self, generalSexClassifier):
		#get correct labels from dictionary in trainY and testY
		trainX = self.spanishTrainData[0]
		trainY = self.getYlabels(self.spanishTrainData[1], 'sex')
		testX = self.spanishTrainData[0]
		testY = self.getYlabels(self.spanishTrainData[1], 'sex')
		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(max_features=1000)),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char", min_df=4, max_features=1000)), 
										("counts", CountVectorizer(max_features=1000)),
										("generalSexClassifier", Classifier(generalSexClassifier)),
										])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='linear')
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		predictY = pipeline.predict(testX)
		return predictY





'''Helper functions'''

	def readData(self,folder, lang):
		X, Y, DOCID = [], [], []
		if os.path.isdir(folder + "/" + lang):
			indir = folder + "/" + lang
			
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
						x, y, docid = self.parseXML(indir + '/' + f, truthDict)
						X.extend(x)
						Y.extend(y)
						DOCID.extend(docid)


		return X,Y,DOCID
					

	def parseXML(self,doc,truthDict):
		xmldoc = minidom.parse(doc)
		xmlID = doc.split("/")[-1][:-4]
		
		itemlist = xmldoc.getElementsByTagName('document')
		x = []
		y = []
		docid = []
		for s in itemlist:
			x.append(self.processTweet(s.firstChild.nodeValue.strip()))
			y.append(truthDict[xmlID])
			docid.append(xmlID)
		
		
		return x,y,docid


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


	
	def evaluation(self,prediction, language):
		
		if language == 'english':
			testY = self.getYlabels(self.englishTestData[1], 'sex')
		
		print("Classification report {} per tweet".format(language))
		print(classification_report(prediction, testY))
		counter = defaultdict(list)
		count = Counter()
		goldLabels = {}
		for i,p,y in zip(self.englishTestData[2],prediction,testY):
			counter[i].append(p)
			goldLabels[i] = y
		
		keys, predict, gold = [], [], []
		for key in counter:
			mc = Counter(counter[key])
			label = mc.most_common(1)[0][0]
			keys.append(key)
			predict.append(label)
			gold.append(goldLabels[key])
			print(key, label, goldLabels[key])

		print("Classification report {} per author".format(language))
		print(classification_report(gold, predict))
		




	def getYlabels(self, y, value):
		
		return [row[value] for row in y]
			
t = Trainer(sys.argv)
