from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import cluster
from sklearn.metrics import completeness_score, v_measure_score, classification_report, accuracy_score, homogeneity_score,confusion_matrix,homogeneity_completeness_v_measure,adjusted_rand_score
import sys
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from collections import defaultdict, Counter

class KmeansClassifier:
	def __init__(self):
		pass

	def read_corpus(self,corpus_file, use_sentiment):
	    documents = []
	    labels = []
	    bigrams = False
	    #stemmer = PorterStemmer()
	    stemmer = SnowballStemmer("english", ignore_stopwords=False)
	    wordnet_lemmatizer = WordNetLemmatizer()
	    with open(corpus_file, encoding='utf-8') as f:
	        for line in f:
	            tokens = line.strip().split()
	            t = [stemmer.stem(token) for token in tokens[3:]]
	            #t = [wordnet_lemmatizer.lemmatize(token) for token in tokens[3:]]
	            
	            if bigrams:   
	                documents.append(list(zip(t, t[1:])))
	            else:
	                documents.append(t)
	            labels.append( tokens[0] )
	    
	    return documents, labels
    
	# a dummy function that just returns its input
	def identity(self,x):
	    return x




	def train(self,argv):
		testmode = False #seperate testfile or do cross validation

		if len(argv) == 2:
		    trainfile = argv[1]
		elif len(argv) == 3:
		    testmode = True
		    trainfile = argv[1]
		    testfile = argv[2]
		else:
		    exit("Use assignment.py trainfile <testfile> <Optional>")


		# X and Y are the result of the read corpus function. X is a list of all documents that are tokenized and Y is a list of all labels
		# The use_sentiment boolean can be changed to use the categories(False) or the polarity(True)
		X, Y = self.read_corpus(trainfile, use_sentiment=True)

		# we use a dummy function as tokenizer and preprocessor,
		# since the texts are already preprocessed and tokenized.
		vec = TfidfVectorizer(preprocessor = self.identity, tokenizer = self.identity,sublinear_tf=True)
		
		km = Pipeline( [('vec', vec),
                            ('cls', cluster.KMeans(n_clusters=6, n_init=10, verbose=1))] )
		
		labels_pred = km.fit_predict(X,Y)
		labels_true = Y

		c = defaultdict(list)
		#calculate confusion matrix
		for pred,true in zip(labels_pred,labels_true):
			c[pred].append(true)

		label = {}
		for key in c:
			count = Counter(c[key])
			label[key] = count.most_common(1)[0][0]

		labels_pred = [label[l] for l in labels_pred]
		labels = list(set(label.values()))
		print(labels)
		

		print("Homogeneity: %0.3f" % homogeneity_score(labels_true, labels_pred))
		print("Completeness: %0.3f" % completeness_score(labels_true, labels_pred))
		print("V-measure: %0.3f" % v_measure_score(labels_true, labels_pred))
		print("Adjusted Rand-Index: %.3f" % adjusted_rand_score(labels_true, labels_pred))
		print(confusion_matrix(labels_true, labels_pred, labels=labels))
	
K = KmeansClassifier()
K.train(sys.argv)






