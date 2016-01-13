from sklearn.base import BaseEstimator, TransformerMixin
import nltk
class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""

	def fit(self, x, y=None):
		return self

	def transform(self, tweets):

	#length is the length of the tweet rounded off to nearest 5
		return [{'length': 1,
				'avg_word_length': 1}
				for t in tweets]
		return [{'length': int(3 * round(float(len(t))/5)),
				'avg_word_length': self.avg_word_length(t)}
				for t in tweets]

	def avg_word_length(self, tweet):
		lengthWord = 0
		words = tweet.split()
		for word in words :
			lengthWord =+ len(word)
		return lengthWord / len(words)


class Classifier(BaseEstimator, TransformerMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
       
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [value for value in list(self.classifier.predict(texts))]
        



class CountCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [[sum(c.isupper() for c in text)] for text in texts]

class CountWordCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital words from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital words in
        :returns: list of counts for each text
        """
        return [[sum(w.isupper() for w in nltk.word_tokenize(text))]
                for text in texts]