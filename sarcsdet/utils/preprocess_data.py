import nltk
import re
import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem


class Preprocessor():
    def __init__(self, lang):
        if lang.lower() not in ['en', 'ru']:
            raise ValueError('Wrong lang argument. Must be en or ru')
        
        self.lang = lang.lower()
        nltk.download('stopwords')
        
        if self.lang == 'en':
            self.stem = stem = nltk.stem.PorterStemmer()
            # self.stem = nltk.stem.SnowballStemmer('english')
            self.stopwords = stopwords.words('english')
        else:
            self.stem = Mystem()
            self.stopwords = stopwords.words('russian')
    
    def preprocess_text(self, text):
        if self.lang == 'en':
            text = re.sub(r'[^\w\s\n]', '', str(text))
            tokens = ''.join(self.stem.stem(text))
            tokens = word_tokenize(tokens)
            tokens = [
                token for token in tokens 
                if token not in self.stopwords and token not in string.punctuation
            ]
            text = ' '.join(tokens)
            return text
        
        else:
            text = text.lower()
            text = re.sub(r'[^\w\s\n]', '', text)
            tokens = ''.join(self.stem.lemmatize(text))
            tokens = word_tokenize(tokens)
            tokens = [
                token for token in tokens 
                if token not in self.stopwords and token not in string.punctuation
            ]
            text = ' '.join(tokens)
            return text
