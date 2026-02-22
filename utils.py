import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ✅ Safe download (only if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)
