import pandas as pd # Used for reading the csv data
import nltk # Used for removing stop words from messages
from nltk.corpus import stopwords

df = pd.read_csv("spam.csv", encoding = 'latin-1')

# 5572 row. 2 columns. 1-ham/spam 2-message
#df.info()

# There are some unwanted extra columns in the data file. To remove them,
df = df.iloc[:, :2]
#print(df)

labels = df["v1"] # Get labels
messages = df["v2"] # Get messages

# Todo: How many records we have?

# Get english stopwords list from nltk library. 179 stop words
sw_nltk = stopwords.words('english')
print(sw_nltk)

# Remove stop words