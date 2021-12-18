import pandas as pd # Used for reading the csv data
import nltk # Used for removing stop words from messages
from nltk.corpus import stopwords
import string # For punctuation

df = pd.read_csv("spam.csv", encoding = 'latin-1')

#df.info()

# There are some unwanted extra columns in the data file. To remove them,
df = df.iloc[:, :2]
#print(df)

labels = df["v1"] # Get labels
messages = df["v2"] # Get messages
# print(labels)
# print(messages)

# Todo: How many records we have?

# Get english stopwords list from nltk library. 179 stop words
#stopwords_array = stopwords.words('english')
#print(stopwords_array)

# STEP: Data preparation
# Remove stop words from messages
# Remove punctuation

stopwords = stopwords.words("english")
punctuations = string.punctuation

stopwords_removed_messages = []
for message in messages:
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
    stopwords_removed_messages.append(new_message)
    
print(messages[15])
print(stopwords_removed_messages[15])
