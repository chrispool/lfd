

class AuthorProfiler():
	"""Author profiler by Chris Pool"""
	
	def __init__(self):
		pass

	def readCorpus(self):
		pass

	def trainNL(self):
		Xtrain, Ytrain, Xtest, Xtrain = self.readCorpus('NL')
		#classify
		sexClassifier = self.trainSexClassifier(Xtrain, Ytrain)

		#evaluate test
		Yprediction = sexClassifier.classifier.predict(Xtest)
		self.evaluate(Yprediction,Ytest)
	
	def trainEN(self):
		pass

	def trainIT(self):
		pass

	def trainEN(self):
		pass

	def trainSexClassifier(self, Xtrain, Ytrain):
		tfidf = True
		if tfidf:
			vec = TfidfVectorizer(sublinear_tf=True, max_df=0.05)
		else:
			vec = CountVectorizer()
		
		classifier = Pipeline( [('vec', vec),('cls', svm.SVC(kernel='rbf', C=1.0, gamma=0.9))] )

		# Train the classifier
		return classifier.fit(Xtrain, Ytrain)


	def loadClassifiers(self):
		pass

	def saveClassfiers(self):
		pass

	def classifyTweet(self):
		pass




AP = AuthorProfiler()

		