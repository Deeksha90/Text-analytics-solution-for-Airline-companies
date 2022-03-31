
# Importing all relevant libraries here
import pandas as pd
from collections import Counter
import operator
import nltk
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
import warnings
import re
import collections
warnings.filterwarnings('ignore')

def word_selection(words):
    return dict([(word,True) for word in words])

# Read the csv; Csv was created using MongoDB connection in the other script
airline_df = pd.read_csv('airline_data.csv')
# Select columns for analysis
review_data = airline_df[['reviewcontent','airlinename','authorname','rating_overall','recommended']]

# Printing the number of data points in the dataset
print("The total number of reviews in dataset are: ", len(airline_df))
# For recommended flag we consider that positive and for not recommended we take it as negative feedback
positive_rec=[]
negative_rec= []

for row in range(0,len(review_data)):
    # print(review_data.iloc[row]['recommended'])
    review_temp = ""
    review_temp = review_data.iloc[row]['reviewcontent']
    # Remove punctuation
    review_temp = re.sub(r'[^\w\s]','',review_temp)
    # Review unigrams
    review_temp = " ".join(word for word in review_temp.split()
                      if len(word)>1)
    # Stopword Removal
    review_temp = " ".join(word for word in review_temp.split()
                if not word in stopwords.words('english'))
    # Stemming
    stemmer = SnowballStemmer('english')
    review_temp = " ".join([stemmer.stem(word)
                for word in review_temp.split()]
                )
    ngramsize = 2
    if review_data.iloc[row]['recommended'] == 1:
      if ngramsize > 1:
          review_temp = [word for word in ngrams(review_temp.split(),ngramsize)]
          review_pos = (word_selection(review_temp),'rec')
      else:
            review_pos = (word_selection(review_temp.split()),'rec')
      positive_rec.append(list(review_pos))

    elif review_data.iloc[row]['recommended'] == 0:
      if ngramsize > 1:
          review_temp = [word for word in ngrams(review_temp.split(),ngramsize)]
          review_neg = (word_selection(review_temp),'not_rec')
      else:
          review_neg = (word_selection(review_temp.split()),'not_rec')
      negative_rec.append(list(review_neg))

print("Number of Positive Reviews", len(positive_rec))
print("Number of Negative Reviews", len(negative_rec))

# Splitting Train and Test data
train_data = positive_rec[:14000] + negative_rec[:14000]
test_data = positive_rec[14000:] + negative_rec[14000:]
print("Length of Train Data: ", len(train_data))
print("Length of Test Data: ", len(test_data))

# Train Naive Bayes Model for classifying Positive and Negative Feedback
classifier=NaiveBayesClassifier.train(train_data)

# actual_label will include the original pos/neg classification from raw data
# predicted_label will include the predicted classification results
actual_label = []
predicted_label = []

# Put the predicted class in the predicted_label
results = []
label_dict = {'rec':1,'not_rec':0}
for i,(input_cols, label) in enumerate(test_data):
    actual_label.append(label_dict[label])
    observed = classifier.classify(input_cols)
    predicted_label.append(label_dict[observed])
    results.append(observed)

# Performance evaluation
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(actual_label, predicted_label)
sns.heatmap(cm, annot=True)
f_score = f1_score(actual_label, predicted_label)

print("The F-score of the prediction: ", f_score)