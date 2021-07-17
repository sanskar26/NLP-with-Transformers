"""
Exploratory Data Analysis and Preprocessing 

Training/Validation Split

Loading Tokenizer and Encoding our Data

Setting up DistilBERT Pretrained Model

Creating Data Loaders

Setting Up Optimizer and Scheduler

Defining our Performance Metrics

Creating our Training Loop

Loading and Evaluating our Model
"""


#!pip install transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import transformers
from transformers import DistilBertTokenizer,DistilBertModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import random


if torch.cuda.is_available():
  device= torch.device("cuda")
  print("There is %d gpu device available"%torch.cuda.device_count() if torch.cuda.device_count()==1 else "There are %d gpu devices available"%torch.cuda.device_count())
  print(f"Using Gpu :{torch.cuda.get_device_name(0)}")

else:
  device="cpu"
  print("Gpu device Not Found, Using Cpu Instead")

df= pd.read_csv("/content/drive/MyDrive/IMDB Dataset.csv") # feed in the path of data file

# idx=[]
# for i in range(25000):
#   idx.append(random.randint(1,50000))

# df= df.iloc[idx,:].copy()
# df.reset_index(drop=True,inplace=True)

df["sentiment"]= pd.Series(np.where(df["sentiment"]=="positive",1,0))

# ax= sns.countplot(df.sentiment)
# plt.xlabel('review score')
# plt.show()

"""### Before tokenizing our text, we will perform some slight processing on our text including removing entity mentions.The level of processing here is much less than in previous approachs because BERT was trained with the entire sentences"""

def re_preprocessing(text):
  # Isolate and remove punctuations except '?'
  text= re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
  text= re.sub(r'[^\w\s\?]', ' ', text)
  
  text = re.sub(r'(@.*?)[\s]'," ",text)
  text= re.sub(r"\s+"," ",text).strip()

  return text 

df["review"]= df["review"].apply(lambda x: re_preprocessing(x))


"""Because we are considering the uncased model, the sentence was lowercased first.
In Bert model, Word piece tokenizer is used which splits "gpu" into known subwords: ["gp" and "##u"].
"##" means that the rest of the token should be attached to the previous one, without space (for decoding or reversal of the tokenization)
"""

# sns.distplot(df["review"].apply(lambda x: len(x.split())),bins=40)
# plt.xlabel("Review Length")
# plt.ylabel("Number of Reviews")
# plt.xlim([0,512])

df["review"]= df["review"].apply(lambda x: " ".join(x.split()[:400]))
max_len = df["review"].apply(lambda x: len(x.split())).max()
# print(max_len)

Max_Len= 400
model_name="distilbert-base-uncased"
tokenizer= DistilBertTokenizer.from_pretrained(model_name,do_lower_case=True)


class Final_embedding_dataset():
    
  def __init__(self,reviews,targets,tokenizer,max_len):
    
    self.reviews= reviews
    self.targets= targets
    self.tokenizer= tokenizer
    self.max_len= max_len
    
  def __len__(self):
    return len(self.reviews)

  def __getitem__(self,item):

    review= self.reviews[item]
    target= self.targets[item]

    encoding = self.tokenizer.encode_plus(review,add_special_tokens=True, padding="max_length",truncation=True, 
                                 return_tensors="pt",verbose=False,max_length=self.max_len,return_attention_mask=True)
    
# return_tensors="pt", gives us the embedding in form of tensors.
# tensors should be flattened.
    
    return {"review":review,
      "input_ids":encoding["input_ids"].flatten(),
      "attention_mask":encoding["attention_mask"].flatten(),
      "target": torch.tensor(target,dtype= torch.long)}

"""
Attention_mask is all about: it points out which tokens the model should pay attention to and which ones it should not (because they represent padding in this case).
token type IDs (also called segment IDs). They are represented as a binary mask identifying the two types of sequence in the model.
positions_ids is used to identify each token's position in the list of tokens

The default positional encoding is absolute(generally used in BERT), whereas some of the model us sinusodial position encoding also.
"""

df_train,df_test= train_test_split(df,test_size=0.2,random_state=1)
# print(df_train.shape,df_test.shape)

"""
Data Loader function from pytorch library could take in a map based dataset or a iterable dataset,
i have used the map based one using the magic methods, __len__ and __getitem__ to create a map based iteratable object
"""

def create_data_loader(df,batch_size,tokenizer,max_len):
  dataset= Final_embedding_dataset(df["review"].to_numpy(),df["sentiment"].to_numpy(),tokenizer,Max_Len)

  return DataLoader(dataset,batch_size=batch_size,num_workers=2)


Batch_Size= 16

train_data_loader= create_data_loader(df_train,Batch_Size,tokenizer,Max_Len)
test_data_loader= create_data_loader(df_test,Batch_Size,tokenizer,Max_Len)

#Experimentation:-
data= next(iter(train_data_loader))
print(data["input_ids"].shape)
print(data["attention_mask"].shape)
print(data["target"].shape)

model= AutoModel.from_pretrained(model_name)
model.config.hidden_size

class SentimentClassifier(nn.Module):
  def __init__(self,n_classes,model):
    super(SentimentClassifier,self).__init__()
    self.model = model
    self.dropout= nn.Dropout(p=0.3)
    self.output= nn.Linear(self.model.config.hidden_size,n_classes)

  def forward(self,input_ids,attention_mask):
    output= self.model(input_ids=input_ids,attention_mask=attention_mask)
    class_output= self.dropout(output["pooler_output"])

    return self.output(class_output)

n_classes= 2
classifier_model= SentimentClassifier(n_classes,model)
classifier_model= classifier_model.to(device)

EPOCHS= 10 

optimizer= AdamW(classifier_model.parameters(),lr=3e-5, correct_bias=False) # in bert actual implementation correct_bias is False
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps= len(train_data_loader)*EPOCHS) # num_steps= num_minibatch*Epoch

loss_function= nn.CrossEntropyLoss().to(device)

# helper function for training. Basically a function for training step- one epoch.

def train_one_epoch(model,data_loader,loss_function,optimizer,device,scheduler,n_examples):

  model= model.train()

  losses=[]
  correct_predictions= 0

  for d in data_loader:
    # input_ids= d.get("input_ids").to(device)
    # attention_mask= d.get("attention_mask").to(device)
    # targets= d.get("target").to(device)

    input_ids, attention_mask, targets = tuple(t.to(device) for t in d)[1:]

    outputs= model(input_ids,attention_mask)
    
    _,preds= torch.max(outputs,dim=1)
    loss= loss_function(outputs,targets)

    correct_predictions+= torch.sum(preds==targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # is necessary to ensure that the gradients doesn't blow up
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    accuracy= correct_predictions/n_examples

  return accuracy,np.mean(losses)

def eval_model(model,data_loader,loss_function,device,n_examples):

  model= model.eval()

  losses=[]
  correct_predictions= 0

  with torch.no_grad():
    for d in data_loader:
      
       input_ids, attention_mask,targets = tuple(t.to(device) for t in d)[1:]
     
      outputs= model(input_ids,attention_mask)
    
      _,preds= torch.max(outputs,dim=1)
      loss= loss_function(outputs,targets)

      correct_predictions+= torch.sum(preds==targets)
      losses.append(loss.item())
    
      accuracy= correct_predictions.double()/n_examples

  return accuracy,np.mean(losses)

#call model using classifier_model
# writing the training loop

history=defaultdict(list)
best_accuracy=0

for epoch in range(EPOCHS):
  print(f"EPOCH: {epoch+1}/{EPOCHS}")
  print("---"*10)

  train_acc,train_loss= train_one_epoch(classifier_model,train_data_loader,loss_function,optimizer,device,scheduler,len(df_train))
 
  print(f"Accuracy: {train_acc} Train loss: {train_loss}")
  
  test_acc, test_loss = eval_model(classifier_model,test_data_loader,loss_function,device, len(df_test) )

  print(f"test_accuracy: {test_acc} Test loss: {test_loss}")

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['test_acc'].append(test_acc)
  history['test_loss'].append(test_loss)

torch.save(model.state_dict(),"model.bin")

# plt.plot(np.arange(EPOCHS),history["train_acc"],label="train_accuracy")
# plt.plot(np.arange(EPOCHS),history["test_acc"],label="test_accuracy")

# plt.xlabel("EPOCHS")
# plt.ylabel("Accuracy")
# plt.title("Accuracy Plot")
# plt.ylim([0,1])
# plt.show()


def get_predictions(model,data_loader):

  model= model.eval()

  review_text= []
  predictions= []
  prediction_probs= []
  target_values= []

  with torch.no_grad():
    for d in data_loader:
      input_ids, attention_mask, targets = tuple(t.to(device) for t in d)[1:]

      outputs= model(input_ids,attention_mask)
    
      _,preds= torch.max(outputs,dim=1)
      probs= F.softmax(outputs,dim=1)

      review_text.append(d["review_text"]).cpu()
      predictions.append(preds).cpu()
      prediction_probs.append(probs).cpu()
      target_values.append(targets).cpu()
      

  return review_text, predictions, prediction_probs, target_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)

print(classification_report(y_test,y_pred))
