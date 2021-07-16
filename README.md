# NLP-with-Transformers 
### Learning Project part of Season of Code, IIT Bombay, 2021
### Mentor- Tezan Sahu,Shreya Laddha

  ## Phase 1-
\
The first week started with getting familiar with `pytorch` by implementing basic tensor operations and a feed forward neural network for classification task on CIFAR-dataset. Along, with that we got introduced to a very powerful text processing library in python which is **nltk** , it provides utility functions for data preprocessing/cleaning task such as removing stopwords,punctuations, Stemming and Lemmatization task , tokenisation of corpus, etc.
\
<br/>
Pytorch being more pythonic, involves Object Oriented Prograaming Paradigm and hence the implementation of the neural Architecture was more intuitive and expressive. Since a computer program cannot take string/word as input , hence the corpus of text used in training must be converted into numeric form which is done using word vector or word embedding. 
\
<br/>
## Very First Implementation:
\
We were supposed to code a feed forward neural net model for Sentiment analysis( Classification Task) on [IMDB moview review dataseet](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from kaggle. I started by first importing and conducting Exploratory data Analysis(EDA) and got the gist of the data distribution. Then used `nltk` and `re` library for preprocessing task, Started with a 2 hidden layer neural net with 500 neurons each and used the cross entropy loss function and Adaptive moment estimation(adam) optmiser for back propagation of the loss function updating the parameters of the network. The f1 score was not too pleasing, obviously it required _hyperparameter tuning_, I finally resorted to 3 hidden layer model with a **softmax** activation function in the output layer. 
\
<br/>
Keeping the preprocessing process same, made changes in the nn Module sub-class's computation graph to incorporate GRU and LSTM architecture which solve the bottleneck of RNNs, i.e. Long term dependency Integration through Gated Mechanism which restricts unncessary information and passes relevant information.
<br/>
## Phase 2-
Got introduced to State-of-the Art Encoder-Decoder based Transformer models.It solved some major bottlenecks in using Recurrent neural architecture like LSTM,GRU. LSTM/GRU, i.e the Gradient vanishing and EXploding and one shot parallel data feeding. The mathematics behind self attention mechanism and it's importance is really useful not only for NLP tasks but also vision and other downstream tasks. Going Ahead, I learnt about different architectures like BERT which is a only Encoder transformer. There have been lots of other architecture created out of BERT naming a few-  `BERT`, `RoBERTa`, `XLNet` & `DistilBERT`. DistilBERT is a distiled version of BERT with almost half the parameters but fast and gives similar types of results. 

### Implementation:- 
* Created a Sentiment Classifier with BERT architecture(* `bert-base-uncased`)
* With the BERT architecture in place, replaced the Model and Tokenizer with:
  * `distilbert-base-uncased` (DistilBERT)

The ` HuggingFace` library proviides various open-source pre-trained models for quick integration with easy to use API and highly user-friendly. Transformers package Documentation was really helpful while using the Transformer models for transfer learning task(Fine tuning)

## Phase 3-

* Got Introduced to Text Generation using Transformers
* We explored State-of-the-art OPEN AI's `GPT-2` and Google's `T5` language model for auto-regressive text generation

Beam search will always find an output sequence with higher probability than greedy search, but is not guaranteed to find the most likely output. The most common n-grams penalty makes sure that no n-gram appears twice by manually setting the probability of next words that could create an already seen n-gram to $0$.
\
<br/>
There are several methods to evaluate Text Generation Models like BLEU score, ROUGE Score which can be easily implementated for any NLP task 
<br/>
* We compared GPT-2 and T5 model for Text Generation on a custom manually scrapped dataset
and evaluated their preformance using `BLEU` scores, codes of which can be found in the phase 3 folder


The `Articles_data_SOC.xlsx` file contains the 10 text articles belonging to 3 different domains- Finance, Scientific Jouranls, Entertainment. The model is feed with the first sentence of each article to produce the whole article in Auto-regressive fashion.The respective BLEU scores were
calculated for each article in each domain as well as an overall BLEU score for each domain.The BLEU scores for each domain were saved and are uploaded. 

Based on the results, we conclude that the GPT-2 model works best in the "Finance" domain whereas the T5 model preforms the
best in the "Music/Entertainment" domain.


## Concluding Remarks:

I'd like to thank my mentor Tezan Sahu for helping me throughout the project and SOC team for providing students 
with this mean to invest their time on learning something new and interesting. 
I enjoyed the project and would work on the implementations which were left.
Thank you guys!
