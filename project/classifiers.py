from sklearn.pipeline import FeatureUnion
from transformers import *
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from collections import Counter


class SexClassifiers():
	

	def trainEnglishSexClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.englishTrainData[0]
		trainY = self.getYlabels(self.englishTrainData[1], 'sex')
		

		combined_features = FeatureUnion([
										("tfidf", TfidfVectorizer()),							
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char")), 
										("hashtags", CountHashtags()),
										("mentions", CountMentions()),
										("english", English()),
										#("countRepeatingLetters", RepeatingLetters()),
										],transformer_weights={
											'english': 1,
											'tfidf': 2,
											'ngrams': 2,
											'hashtags': 1,
											'mentions': 1,
											'countRepeatingLetters' : 1
        								})

		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.LinearSVC()
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)

		return pipeline

	
	def trainDutchSexClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.dutchTrainData[0]
		trainY = self.getYlabels(self.dutchTrainData[1], 'sex')
		
		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer()),								
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char")), 
										("counts", CountVectorizer()),	
										],transformer_weights={
											'tfidf': 2,
											'counts': 1,
											'ngrams': 2,

        								})
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.LinearSVC()
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		return pipeline


	def trainItalianSexClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.italianTrainData[0]
		trainY = self.getYlabels(self.italianTrainData[1], 'sex')

		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer()),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char")), 
										("counts", CountVectorizer()),
										#("generalSexClassifier", Classifier(generalSexClassifier)),
										("latin", Latin()),	
										],transformer_weights={
											'latin': 1,
											'tfidf': 2,
											'ngrams': 2,

        								})
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.LinearSVC()
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		return pipeline

	def trainSpanishSexClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.spanishTrainData[0]
		trainY = self.getYlabels(self.spanishTrainData[1], 'sex')

		

		combined_features = FeatureUnion([("tfidf", TfidfVectorizer()),
										("ngrams", TfidfVectorizer(ngram_range=(3, 3), analyzer="char")), 
										("counts", CountVectorizer()),
										],transformer_weights={
											'latin': 1,
											'tfidf': 2,
											'counts': 1,
											'ngrams': 2,

        								})
		classifier = svm.LinearSVC()
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		return pipeline

'''Age classifiers'''
class AgeClassifiers():

	def trainEnglishAgeClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.englishTrainData[0]
		trainY = self.getYlabels(self.englishTrainData[1], 'age')

	
		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.05 )),
										("repeatingLetters", RepeatingLetters()),
										("countsWordCaps", CountWordCaps())
										])
		
		classifier = svm.SVC(kernel='rbf', C=1.0, gamma=0.9)
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
	
		return pipeline


	

	def trainSpanishAgeClassifier(self):
		#get correct labels from dictionary in trainY and testY
		trainX = self.spanishTrainData[0]
		trainY = self.getYlabels(self.spanishTrainData[1], 'age')
		
		


		combined_features = FeatureUnion([("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.05 )),
										("repeatingLetters", RepeatingLetters()),
										("countsWordCaps", CountWordCaps())
										])
		
		X_features = combined_features.fit(trainX, trainY).transform(trainX)
		classifier = svm.SVC(kernel='rbf', C=1.0, gamma=0.9)
		pipeline = Pipeline([("features", combined_features), ("classifier", classifier)])
		pipeline.fit(trainX, trainY)
		
		return pipeline