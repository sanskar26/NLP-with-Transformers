#!pip install nltk
#!pip install pyspellchecker

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download(["punkt","wordnet"])
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim import corpora
#from sklearn.feature_extraction.text import TfidfVectorizer
#from spellchecker import SpellChecker


df= pd.read_csv("IMDB Dataset.csv") 

#print(df["sentiment"].unique())

"""You will find the sentiment is either postive or negative, we need to convert these values to binary values to feed it into our model"""

df["sentiment"]= pd.Series(np.where(df["sentiment"]=="positive",1,0))

#Plotting the Number of samples belonging to each class
df["sentiment"].value_counts().plot.bar(title= "Sentiment Distribution in the dataset")
plt.xlabel("Sentiments")
plt.show()

"""
## Text Preprocessing:-

1. Removing punctuations,symbols,html tags
2. lemmitisation, Spelling Correction
3. Tokenisation

Why do you need to preprocess this text?
Not all the information is useful in making predictions or doing classifications. Reducing the number of words will reduce the input dimension to your model. 
The way the language is written, it contains lot of information which is grammar specific.
Thus when converting to numeric format, word specific characteristics like capitalisation, punctuations, suffixes/prefixes etc. are redundant. 
Which methods to apply and which ones to skip depends on the problem at hand.
"""

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]+"

"""Lowercasing and removing all sorts of punctuations including html tags, urls"""


def clean_text(text):
  text = re.sub(TEXT_CLEANING_RE,' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    tokens.append(token)
  return " ".join(tokens)

df["review"]= df["review"].apply(lambda text: clean_text(text))

lemma= WordNetLemmatizer()

def lemmatize_text(text):
  return " ".join([lemma.lemmatize(word) for word in text.split()])


df["lemmatised_text"]= df["review"].apply(lambda text: lemmatize_text(str(text)))
df["tokenized_text"]=df["lemmatised_text"].apply(lambda text: word_tokenize(text))

#def stem(text):
  #ps=PorterStemmer()
  #return " ".join([ps.stem(word) for word in text.split()])

#df["review"]= df["review"].apply(lambda text: stem(text))

#spell= SpellChecker()
#def spellcheck(text):
  #misspelled_word= spell.unknown(text.split())
  #return " ".join([spell.correction(word) if word in misspelled_word else word for word in text.split()])

#df["review"]= df["review"].apply(lambda text: spellcheck(text))

"""Splitting into train and test set:-"""

X_train, X_test, Y_train, Y_test = train_test_split(df['tokenized_text'], df['sentiment'],shuffle=True,test_size=0.2,random_state=10)

X_train = X_train.reset_index()
X_test = X_test.reset_index()
Y_train = Y_train.to_frame()
Y_train = Y_train.reset_index()
Y_test = Y_test.to_frame()
Y_test = Y_test.reset_index()

if torch.cuda.is_available():
  device= torch.device("cuda")
else:
  device= torch.device("cpu")
print(device)

# Function to return the dictionary
def make_dict(df_small):
  vocab_dict = corpora.Dictionary(df_small["tokenized_text"])
  return vocab_dict

# Make the dictionary:-
vocab_dict = make_dict(df_small)

Vocab_size = len(vocab_dict)
# Function to make bow vector to be used as input to network
def make_bow_vector(vocab_dict, sentence):
  vec = torch.zeros(Vocab_size, dtype=torch.float64, device=device)
  for word in sentence:
      vec[vocab_dict.token2id[word]] += 1
  return vec.view(1, -1).float()

def make_target(label):
    if label == 1:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([0], dtype=torch.long, device=device)

"""
Defining the Feed forward model.
I have taken  hidden layer with different neurons. And to start with have considered Relu activation function
"""

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        
        # First Hidden Layer:-
        self.f1 = nn.Linear(input_dim, hidden_dim) 
        # Adding Non-linearity 
        self.relu1 = nn.ReLU()

        # Second Hidden Layer:-
        self.f2 = nn.Linear(hidden_dim, hidden_dim) 
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Third Hidden Layer:-
        self.f3 = nn.Linear(hidden_dim, hidden_dim) 
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Output Layer
        self.f4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
       
        # Linear function 1
        out = self.f1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.f2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3
        out = self.f3(out)
        # Non-linearity 3
        out = self.relu3(out)

        #output
        out = self.f4(out)

        return F.softmax(out, dim=1)

    

input_dim= Vocab_size
hidden_dim= 500
output_dim=2
num_epochs=10
learning_rate=0.1
batch_size=100

# RUN TRAINING AND TEST

ff_nn_bow_model= FeedforwardNN(input_dim,hidden_dim,output_dim)
ff_nn_bow_model.to(device)

loss_function= nn.CrossEntropyLoss()
optimizer= optim.Adam(ff_nn_bow_model.parameters(), lr=learning_rate)


#starting training
for epoch in range(num_epochs):
  train_loss=0
  for index, row in X_train.iterrows():
    #Clearing the accumulated gradients
    # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
    optimizer.zero_grad()

    # Make the bag of words vector for stemmed tokens 
    bow_vec = make_bow_vector(vocab_dict, row["tokenized_text"])
      
    #Forward pass to get output
    probs = ff_nn_bow_model(bow_vec)

    #Getting the target label
    target = make_target(Y_train["sentiment"][index])

    #Calculate loss
    loss = loss_function(probs,target)
    
    #Accumulating the loss over time
    train_loss+=loss.item()
    
    #Getting gradients wrt paramters:-
    loss.backward()

    #Updating Paramters
    optimizer.step()

  print(str(epoch+1)+ " " + str(train_loss/len(X_train)))
  print("\n")
  

#Evaluation
bow_ff_nn_predictions = []
original_lables_ff_bow = []

with torch.no_grad():
    for index, row in X_test.iterrows():
        bow_vec = make_bow_vector(vocab_dict, row)
        probs = ff_nn_bow_model(bow_vec)
        bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
        original_lables_ff_bow.append(make_target(Y_test["sentiment"][index]).cpu().numpy()[0])

print(classification_report(original_lables_ff_bow,bow_ff_nn_predictions))
