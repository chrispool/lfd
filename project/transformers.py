from sklearn.base import BaseEstimator, TransformerMixin

class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""

	def fit(self, x, y=None):
		return self

	def transform(self, tweets):

	#length is the length of the tweet rounded off to nearest 5
		return [{'length': int(3 * round(float(len(t))/5)),
				'avg_word_length': self.avg_word_length(t)}
				for t in tweets]

	def avg_word_length(self, tweet):
		lengthWord = 0
		words = tweet.split()
		for word in words :
			lengthWord =+ len(word)
		return lengthWord / len(words)

