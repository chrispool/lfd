from sklearn.base import BaseEstimator, TransformerMixin
import nltk



class SexClassifier(BaseEstimator,TransformerMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prediction = []
        for p in list(self.classifier.predict(X)):
            if p == 'M':
                prediction.append(1)
            else:
                prediction.append(2)

        test = [ [p] for p in prediction]
        return(test)
        #return [[sum(c.isupper() for c in text)] for text in X]

class Classifier(BaseEstimator,TransformerMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prediction = []
        for p in list(self.classifier.predict(X)):
            if p == 'M':
                prediction.append(1)
            else:
                prediction.append(2)

        test = [ [p] for p in prediction]
        return(test)
        #return [[sum(c.isupper() for c in text)] for text in X]

class Postags(BaseEstimator, TransformerMixin):
    """ Model that add postag """

    def fit(self, X, y=None):
        return self

    def postag(self, tweet):
        postags = nltk.pos_tag(nltk.word_tokenize(tweet))
        return " ".join([tag for token, tag in postags])

    def transform(self, texts):    
        result = [ self.postag(text) for text in texts]
        return result

class CountPunctuation(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        punctuation=['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'"]
        result = [[len([c for c in text if c in punctuation])] for text in texts]
        return result


class CountHashtags(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def countTags(self, tweet):
        c = 0
        for word in nltk.word_tokenize(tweet):
            if word[0] == '#':
                c =+ 1
        return c

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [[self.countTags(text)] for text in texts]

class Latin(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def indicators(self,tweet):
        c = 0
        for word in nltk.word_tokenize(tweet):
            if word[-1] == 'a':
                c =+ 1
        return c

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [[self.indicators(text)] for text in texts]


class English(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def indicators(self,tweet):
        femaleWords = ['boyfriend', 'man', 'husband']
        maleWords = ['girlfriend', 'wife', 'girl', 'mate']
       
        for word in nltk.word_tokenize(tweet):
            if word in femaleWords:
                return 1
            elif word in maleWords:
                return 2
        
        return 0

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """

        return [ [self.indicators(text)] for text in texts]

class CountMentions(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def countMentions(self, tweet):
        c = 0
        for word in nltk.word_tokenize(tweet):
            if word == 'AT_USER':
                c =+ 1
        return c

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [[self.countMentions(text)] for text in texts]
       
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

class RepeatingLetters(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def countRepeatingLetters(self, tweet):
        c = 0
        for word in nltk.word_tokenize(tweet):
            for i,ch in enumerate(word):
                if len(word) > i + 3:
                   
                    if word[i+1:i+3] == ch+ch:
                        c =+ 1
                        
        return c

    def transform(self, texts):
        """ transform data
        :texts: The texts to count capital letters in
        :returns: list of counts for each text
        """
        return [[int(self.countRepeatingLetters(text))] for text in texts]

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