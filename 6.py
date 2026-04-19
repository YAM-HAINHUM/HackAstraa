# STEP 1: IMPORT LIBRARIES
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from textblob import TextBlob

# Download required datasets (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# STEP 2: INPUT TEXT
text = "I really love this product! It is amazing and works perfectly."

print("Original Text:", text)

# STEP 3: CLEANING TEXT
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

cleaned_text = clean_text(text)
print("Cleaned Text:", cleaned_text)

# STEP 4: TOKENIZATION
tokens = word_tokenize(cleaned_text)
print("Tokens:", tokens)

# STEP 5: POS TAGGING
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

# STEP 6: REMOVE STOPWORDS
stop_words = set(stopwords.words('english'))

filtered_words = [word for word in tokens if word not in stop_words]
print("After Stopword Removal:", filtered_words)

# STEP 7: STEMMING
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Stemmed Words:", stemmed_words)

# STEP 8: LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Lemmatized Words:", lemmatized_words)

# STEP 9: SENTIMENT ANALYSIS (TEXTBLOB)
analysis = TextBlob(text)

print("Polarity:", analysis.sentiment.polarity)
print("Subjectivity:", analysis.sentiment.subjectivity)

# Classify sentiment
if analysis.sentiment.polarity > 0:
    print("Sentiment: Positive")
elif analysis.sentiment.polarity < 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")