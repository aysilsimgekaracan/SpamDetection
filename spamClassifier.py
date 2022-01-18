import pandas as pd # Used for reading the csv data
from nltk.corpus import stopwords
import string # For punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten

"""
STEP 1

DATA PREPARATION
"""

df = pd.read_csv("spam.csv", encoding = 'latin-1')

#df.info()

# There are some unwanted extra columns in the data file. To remove them,
df = df.iloc[:, :2]

df.columns = ['target', 'message'] # Change column names


# Sets ham to 0, spam to 1
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])

# df['message_length'] = df.message.apply(len)

stopwords = stopwords.words("english")

def data_preparation(message):
    """Removes stopwords and punctuations

    Args:
        message (string): message

    Returns:
        string: new cleaned message
    """
    # messages = df["message"] # Messages column
    punctuations = string.punctuation

    words = []
    for word in message.split():
        word = word.lower()
        if word not in stopwords:
            chars = []
            for char in word:
                if char not in punctuations:
                    chars.append(char)
            
            new_word = "".join(chars)
            words.append(new_word) 
    
    new_message = " ".join(words)
    
    return new_message
    

# Add cleaned_messages to df
df['cleaned_message'] = df.message.apply(data_preparation)

"""
STEP 2

MODELLING
"""
targets = df.target
messages = df.cleaned_message

# Split train and test data
# - train_test_split -
#   - Split arrays or matrices into random train and test subsets
#   - test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#   - random_state: Controls the shuffling applied to the data before applying the split.
#   - stratify: mmm

messages_train, messages_test, targets_train, targets_test = train_test_split(messages, targets, test_size=0.2, random_state=20)
print("Lenght of messages_train: " + str(len(messages_train)))
print("Lenght of messages_test: " + str(len(messages_test)))
print("Lenght of targets_train: " + str(len(targets_train)))
print("Lenght of targets_test: " + str(len(targets_test)))
print("Ratio: " + str(100*len(messages_train)/(len(messages_train) + len(messages_test))))


