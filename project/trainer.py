import os, re, sys
from xml.dom import minidom
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter,defaultdict
from classifiers import AgeClassifiers,SexClassifiers

class Trainer(AgeClassifiers, SexClassifiers):
	def __init__(self, argv):
		
		#seperate train and testset. In this modus the truth files will be written to the test directory
		if len(argv) == 3:

			self.TESTMODE = False
			#read trainingData, list with 3 values, X, Y and DOCID
			self.englishTrainData = self.readData(argv[1],'english')
			self.dutchTrainData = self.readData(argv[1],'dutch')
			self.italianTrainData = self.readData(argv[1],'italian')
			self.spanishTrainData = self.readData(argv[1],'spanish')
			self.allTrainData = [i1+i2+i3+i4 for i1,i2,i3,i4 in zip(self.englishTrainData, self.dutchTrainData, self.italianTrainData, self.spanishTrainData)]
			
			#read all test data
			self.englishTestData = self.readData(argv[2],'english', False)
			self.dutchTestData = self.readData(argv[2],'dutch', False)
			self.italianTestData = self.readData(argv[2],'italian', False)
			self.spanishTestData = self.readData(argv[2],'spanish', False)
			self.allTestData = [i1+i2+i3+i4 for i1,i2,i3,i4 in zip(self.englishTestData, self.dutchTestData, self.italianTestData, self.spanishTestData)]

			
		#only trainset, split the data in train and test. In this modus no thruth files will be written. 
		elif len(argv) == 2:
			
			self.TESTMODE = True
			enData = self.readData(argv[1],'english')
			nlData = self.readData(argv[1],'dutch')
			itData = self.readData(argv[1],'italian')
			spData = self.readData(argv[1],'spanish')
			#combine datasets for training general classifier
			allData = [i1+i2+i3+i4 for i1,i2,i3,i4 in zip(enData, nlData, itData, spData)]
			
			splitpointEN = int(len(enData[0]) * 0.75)
			splitpointNL = int(len(nlData[0]) * 0.75)
			splitpointIT = int(len(itData[0]) * 0.75)
			splitpointSP = int(len(spData[0]) * 0.75)
			splitpointALL = int(len(allData[0]) * 0.75)
			
			self.englishTrainData = [key[:splitpointEN] for key in enData]
			self.dutchTrainData = [key[:splitpointNL] for key in nlData]
			self.italianTrainData = [key[:splitpointIT] for key in itData]
			self.spanishTrainData = [key[:splitpointSP] for key in spData]
			self.allTrainData = [key[:splitpointALL] for key in allData]

			self.englishTestData = [key[splitpointEN:] for key in enData]
			self.dutchTestData = [key[splitpointNL:] for key in nlData]
			self.italianTestData = [key[splitpointIT:] for key in itData]
			self.spanishTestData = [key[splitpointSP:] for key in spData]
			self.allTestData = [key[splitpointALL:] for key in allData]


		else:
			exit("usage trainer.py <trainset> *<testset>")

		print("Data loaded")


		#dictoionary for final results
		self.results = defaultdict(list)

		#train and predict gender for each user	
		self.sexClassifier()
		
		#calculate overal accuracy and classification report for gender classification if there is a gold standard.
		if self.TESTMODE:
			pred, gold = [], []
			for key in self.results:
				pred.append(self.results[key][0][0])
				gold.append(self.results[key][0][1])
			print()
			print("-----------------------{}-----------------------".format("Average"))
			print("Per user")
			print("-----------")
			print()
			print("Accuracy: {}".format(round(accuracy_score(gold, pred),2)))
			print()
			print(classification_report(gold,pred))


		#train and predict age for each user, one of the features is the sex classifier			
		self.ageClassifier()

		#calculate overal accuracy and classification report for age classification if there is a gold standard.
		if self.TESTMODE:
			pred, gold = [], []
			for key in self.results:
				if key[0] == 'english' or key[0] == 'spanish':
					pred.append(self.results[key][1][0])
					gold.append(self.results[key][1][1])
			print()
			print("-----------------------{}-----------------------".format("Average"))
			print("Per user")
			print("-----------")
			print()
			print("Accuracy: {}".format(round(accuracy_score(gold, pred),2)))
			print()
			print(classification_report(gold,pred))


		#write truth file
		if self.TESTMODE == False:
			self.writeTruthFiles(argv[2])



	def ageClassifier(self):
		'''general function that coordinates the age classification'''

		print("Train English classifier for age classification")
		englishAgeClassifier = self.trainEnglishAgeClassifier()

		print("Train Spanish classifier for age classification")
		spanishAgeClassifier = self.trainSpanishAgeClassifier(self.spanishSexClassifier)
		
		#calculate evaluation
		if self.TESTMODE:
			self.evaluationAge(englishAgeClassifier.predict(self.englishTestData[0]), 'english')
			self.evaluationAge(spanishAgeClassifier.predict(self.spanishTestData[0]), 'spanish')
		#No gold label, calculate majority label for truth file
		else:
			self.calculateMajorityLabel(englishAgeClassifier.predict(self.englishTestData[0]), 'english')
			self.calculateMajorityLabel(spanishAgeClassifier.predict(self.spanishTestData[0]), 'spanish')


	def sexClassifier(self):
		'''general function that coordinates the sex classification'''

		#train models for each language using the general sex classifier as feature		
		print("Train English classifier for gender classification")
		self.englishSexClassifier = self.trainEnglishSexClassifier()
		
		print("Train Dutch classifier for gender classification")
		dutchSexClassifier = self.trainDutchSexClassifier()
		
		print("Train Italian classifier for gender classification")
		italianSexClassifier = self.trainItalianSexClassifier()
		
		print("Train Spanish classifier for gender classification")
		self.spanishSexClassifier = self.trainSpanishSexClassifier()
		

		if self.TESTMODE:
			self.evaluationSex(self.englishSexClassifier.predict(self.englishTestData[0]), 'english')
			self.evaluationSex(dutchSexClassifier.predict(self.dutchTestData[0]), 'dutch')
			self.evaluationSex(italianSexClassifier.predict(self.italianTestData[0]), 'italian')
			self.evaluationSex(self.spanishSexClassifier.predict(self.spanishTestData[0]), 'spanish')

		else:
			self.calculateMajorityLabel(self.englishSexClassifier.predict(self.englishTestData[0]), 'english')
			self.calculateMajorityLabel(dutchSexClassifier.predict(self.dutchTestData[0]), 'dutch')
			self.calculateMajorityLabel(italianSexClassifier.predict(self.italianTestData[0]), 'italian')
			self.calculateMajorityLabel(self.spanishSexClassifier.predict(self.spanishTestData[0]), 'spanish')

	def calculateMajorityLabel(self,prediction, language):
		'''Function to calculate what the most frequent label is in a prediction and is used in truth.txt'''
		if language == 'english':
			results = zip(self.englishTestData[2],prediction)

		elif language == 'dutch':
			results = zip(self.dutchTestData[2],prediction)

		elif language == 'italian':
			results = zip(self.italianTestData[2],prediction)

		elif language == 'spanish':
			results = zip(self.spanishTestData[2],prediction)

		predictions = defaultdict(list)
		for user_id, prediction in results:
			predictions[user_id].append(prediction)

		for user in predictions:
			self.results[(language, user)].append(Counter(predictions[user]).most_common(1)[0][0])

	
	def evaluationSex(self,prediction, language):
		'''Evaluate gender classifier in case of gold labels available'''		
		if language == 'english':
			testY = self.getYlabels(self.englishTestData[1], 'sex')
			results = zip(self.englishTestData[2],prediction,testY)

		elif language == 'dutch':
			testY = self.getYlabels(self.dutchTestData[1], 'sex')
			results = zip(self.dutchTestData[2],prediction,testY)

		elif language == 'italian':
			testY = self.getYlabels(self.italianTestData[1], 'sex')
			results = zip(self.italianTestData[2],prediction,testY)

		elif language == 'spanish':
			testY = self.getYlabels(self.spanishTestData[1], 'sex')
			results = zip(self.spanishTestData[2],prediction,testY)
		print()
		print("-----------------------{}-----------------------".format(language))
		print("Per tweet")
		print()
		print("-----------")
		print("Accuracy: {}".format(round(accuracy_score(testY, prediction),2)))
		print()
		print(classification_report(testY,prediction))

		counter = defaultdict(list)
		count = Counter()
		goldLabels = {}
		for i,p,y in results:
			counter[i].append(p)
			goldLabels[i] = y
		
		keys, predict, gold = [], [], []
		for key in counter:
			mc = Counter(counter[key])
			label = mc.most_common(1)[0][0]
			keys.append(key)
			predict.append(label)
			gold.append(goldLabels[key])
			self.results[(language, key)].append((label,goldLabels[key]))

		print("Per user")
		print("-----------")
		print()
		print("Accuracy: {}".format(round(accuracy_score(gold, predict),2)))
		print()
		print(classification_report(gold, predict))
		print()
		

	def evaluationAge(self,prediction, language):
	'''Evaluate age classifier in case of gold labels available'''		
		if language == 'english':
			testY = self.getYlabels(self.englishTestData[1], 'age')
			results = zip(self.englishTestData[2],prediction,testY)

		elif language == 'dutch':
			testY = self.getYlabels(self.dutchTestData[1], 'age')
			results = zip(self.dutchTestData[2],prediction,testY)

		elif language == 'italian':
			testY = self.getYlabels(self.italianTestData[1], 'age')
			results = zip(self.italianTestData[2],prediction,testY)

		elif language == 'spanish':
			testY = self.getYlabels(self.spanishTestData[1], 'age')
			results = zip(self.spanishTestData[2],prediction,testY)

		print()
		print("-----------------------{}-----------------------".format(language))
		print("Per tweet")
		print("-----------")
		print()
		print("Accuracy: {}".format(round(accuracy_score(testY, prediction),2)))
		print()
		print(classification_report(testY,prediction))
		counter = defaultdict(list)
		count = Counter()
		goldLabels = {}
		for i,p,y in results:
			counter[i].append(p)
			goldLabels[i] = y
		
		keys, predict, gold = [], [], []
		for key in counter:
			mc = Counter(counter[key])
			label = mc.most_common(1)[0][0]
			keys.append(key)
			predict.append(label)
			gold.append(goldLabels[key])
			self.results[(language, key)].append((label,goldLabels[key]))
		print("Per user")
		print("-----------")
		print()
		print("Accuracy: {}".format(round(accuracy_score(gold, predict),2)))
		print()
		print(classification_report(gold, predict))
		print()


	'''Helper functions'''
	def writeTruthFiles(self,outputFolder):
		print("Write truth files")
		result = defaultdict(list)
		for key in self.results:
			gender = self.results[key][0]
			if len(self.results[key]) == 2:
				age = self.results[key][1]
			else:
				age = ''
			result[key[0]].append((key[1], gender,age))
		
		
		for language in result:
			self.writeTruthFile(language, outputFolder, result[language])


	def writeTruthFile(self, language, folder, result):
		'''Write truth file to disk'''
		print("writing to file {}".format(folder + "/" + language + "/truth.txt"))
		fobj = open(folder + "/" + language + "/truth.txt", 'w')
		for user_id, gender, age in result:
			fobj.write("{}:::{}:::{} \n".format(user_id, gender, age ))
		fobj.close()

	def readData(self,folder, lang, train=True):
		'''Read all files in folder'''
		X, Y, DOCID = [], [], []
		if os.path.isdir(folder + "/" + lang):
			indir = folder + "/" + lang
			truthDict = {}
			if train:	
				#read truth file
				with open(indir + '/truth.txt', encoding='utf-8') as f:
					for line in f:
						elements = line.strip().split(":::")
						truthDict[elements[0]] = {'sex' : elements[1], 'age' : elements[2], 'f1': elements[3], 'f2' : elements[4], 'f3': elements[5], 'f4' : elements[6], 'f5': elements[7]}

			#read XML files
			for root, dirs, filenames in os.walk(indir):
				for f in filenames:
					if f.endswith('xml'):
						if train == False:
							truthDict[f[:-4]] = {}
						x, y, docid = self.parseXML(indir + '/' + f, truthDict)
						X.extend(x)
						Y.extend(y)
						DOCID.extend(docid)


		return X,Y,DOCID
					

	def parseXML(self,doc,truthDict):
		'''Parse XML and return list with tweets, labels and docids'''
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


	def rpt_repl(self,match):
		return match.group(1)+match.group(1)

	def processTweet(self, tweet):
		'''Pre processing function'''
		#Convert to lower case
		tweet = tweet.lower()
		#Convert www.* or https?://* to URL
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
		#Convert @username to AT_USER
		tweet = re.sub('@[^\s]+','AT_USER',tweet)
		#Convert numbers to NUMBER
		tweet = re.sub('[0-9]+','NUMBER',tweet)
		#Remove additional white spaces
		tweet = re.sub('[\s]+', ' ', tweet)
		
		#floating chars heeeeeeii to heeei
		rpt_regex = re.compile(r"(.)\1{1,}\1{2,}", re.IGNORECASE);
		tweet = re.sub( rpt_regex, self.rpt_repl, tweet )


		#trim
		tweet = tweet.strip('\'"')
		return tweet


	
		


	def getYlabels(self, y, value):
		
		return [row[value] for row in y]
			
t = Trainer(sys.argv)
