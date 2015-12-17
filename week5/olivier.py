from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, adjusted_rand_score, v_measure_score, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
import sys, numpy
from nltk.stem.wordnet import WordNetLemmatizer

def read_corpus(corpus_file):
	wnl = WordNetLemmatizer()
	ngrams = 1
	documents=[]
	labels=[]
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens=line.strip().split()
			lemmedTokens = [wnl.lemmatize(token) for token in tokens[3:]]
			if ngrams == 1:
				documents.append(tokens[3:])
			if ngrams == 2:
				documents.append(zip(lemmedTokens, lemmedTokens[1:]))
			if ngrams == 3:
				documents.append(zip(lemmedTokens, lemmedTokens[1:], lemmedTokens[2:]))
			labels.append(tokens[0])
	return documents, labels

X, Y = read_corpus('all_sentiment_shuffled.txt')#use_sentiment=True
split_point = int(0.5*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x).fit(Xtrain)
trainVec = vec.transform(Xtrain)
km = KMeans(n_clusters=6, n_init=5, verbose=1)#, precompute_distances='auto')
km.fit(trainVec)
testVec = vec.transform(Xtest)
Yguess = km.predict(testVec)

print("Rand index: {}".format(adjusted_rand_score(Ytest,Yguess)))
print("V-measure: {}".format(v_measure_score(Ytest,Yguess)))
print("All three: {}".format(homogeneity_completeness_v_measure(Ytest,Yguess)))
print(confusion_matrix(km.labels_,Yguess))

cm=confusion_matrix(km.labels_, Yguess, labels=list(set(Y)))
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of multi label K-Means classification')
plt.colorbar()
tick_marks = numpy.arange(len(list(set(Y))))
plt.xticks(tick_marks, list(set(Y)), rotation=45)
plt.yticks(tick_marks, list(set(Y)))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()