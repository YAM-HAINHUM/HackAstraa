# STEP 1: IMPORT LIBRARIES
import nltk
import string
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from textblob import TextBlob

# STEP 2: DOWNLOAD REQUIRED DATA
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# STEP 3: INPUT TEXT
text = "I really love this product! It is amazing and works perfectly. The design is beautiful and performance is excellent."

print("Original Text:", text)

# STEP 4: CLEANING TEXT
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

cleaned_text = clean_text(text)
print("Cleaned Text:", cleaned_text)

# STEP 5: TOKENIZATION
tokens = word_tokenize(cleaned_text)
print("Tokens:", tokens)

# STEP 6: REMOVE STOPWORDS
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word not in stop_words]

print("Filtered Words:", filtered_words)

# STEP 7: WORD FREQUENCY VISUALIZATION
word_freq = Counter(filtered_words)

words = list(word_freq.keys())
counts = list(word_freq.values())

plt.figure()
plt.bar(words, counts)
plt.title("Word Frequency Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# STEP 8: STEMMING & LEMMATIZATION
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print("Stemmed:", stemmed_words)
print("Lemmatized:", lemmatized_words)

# STEP 9: SENTIMENT ANALYSIS
analysis = TextBlob(text)

polarity = analysis.sentiment.polarity
subjectivity = analysis.sentiment.subjectivity

print("Polarity:", polarity)
print("Subjectivity:", subjectivity)

# Sentiment Label
if polarity > 0:
    sentiment = "Positive"
elif polarity < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print("Sentiment:", sentiment)

# STEP 10: SENTIMENT VISUALIZATION
labels = ["Polarity", "Subjectivity"]
values = [polarity, subjectivity]

plt.figure()
plt.bar(labels, values)
plt.title("Sentiment Analysis Scores")
plt.ylabel("Score")
plt.show()

# STEP 11: PIE CHART (SENTIMENT)
sentiment_counts = {
    "Positive": 1 if sentiment == "Positive" else 0,
    "Negative": 1 if sentiment == "Negative" else 0,
    "Neutral": 1 if sentiment == "Neutral" else 0
}

plt.figure()
plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
plt.title("Sentiment Distribution")
plt.show()
