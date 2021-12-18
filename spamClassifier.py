import pandas as pd # Used for reading the csv data
from nltk.corpus import stopwords
import string # For punctuation
from sklearn.preprocessing import LabelEncoder

"""
STEP 1

DATA PREPARATION
"""

df = pd.read_csv("spam.csv", encoding = 'latin-1')

#df.info()

# There are some unwanted extra columns in the data file. To remove them,
df = df.iloc[:, :2]

df.columns = ['label', 'message'] # Change column names

# 0 = ham, 1 = spam
encoder=LabelEncoder()
df['label']=encoder.fit_transform(df['label'])

df['message_length'] = df.message.apply(len)


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
print(df.head())

"""
STEP 2

MODELLING
"""
