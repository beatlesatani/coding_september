# sentiment_analysis
this is a repository of sentiment_analysis for September coding challenge.
My sentiment analysis web application takes a sentence as an input, and return the predicted sentiment from "negative,neutral,positive".
This sentiment analysis project consists of two parts.
1. modeling part
2. web application part

In this sentiment_analysis, I used the dataset which were taken from twitter,train.csv,test.csv.

※As a practice of NLP, I worked on the text_classification model in the NLP_practice notebook.

## 1.modeling part
In the model, I adopted BERT model as a word embedding, not word2vec. An embedding is a compressed representation of data such as text or images as continuous vectors in a lower-dimensional space so that ML/DL model can take sentences as inputs. BERT’s bidirectional encoding strategy allows it to ingest the position of a each word in a sequence and incorporate that into that word’s embedding, while Word2Vec embeddings aren’t able to account for word position. In other words, BERT models is able to takes context into the model (ex. minute. this word has 2 meanings. 1.the unit of time.2.small. Usual word embedding methods like word2vec assigned same vector to two-different 'minute'. However in BERT model, it assigned different vectors to two-different 'muinute').
Here,I made three notebooks for making the sentiment_analysis model,

In the first notebook (sentiment_analysis_1.ipynb), I made sentiment analysis model based on the lecture video on Youtube. In this model, I applied pre_trained bert model(preprocessor, encoder). After that, I put embedding inputs into simple Neural Network Layers, and I trained the model with 10 epochs. However the accuracy was lower than 60%, which  was much lower than I expected. As a reason of this score, simple neural network layers were not enough to make good prediction. In addition just applying pre_trained bert encoder were not suitable.
In the second notebook (sentiment_analysis_2.ipynb), I decided to use transformer library from Hugging-face. Hugging Face Transformers is an open-source framework for deep learning created by Hugging Face. It provides APIs and tools to download state-of-the-art pre-trained models and further tune them. From this library, I used pre_train tokenizer named bert-base-cased to set up embedding. In general, Classification model takes two inputs, input_id, attention_mask. 

1.Input_ids
    input_ids are simply the numeric representations of the tokens. Simply single words are converted into numerical value. In addition, there are special tokens exist,[SEP],[CLS],[PAD]. [SEP](token = 102) is added at the end of the sentence. [CLS] (token = 101)is added at the beginning of the sentence. [PAD](token = 0) means padding.

2. attention_mask
  Attention_mask is useful when we add padding to the input tokens. The attention mask tells us which input_ids correspond to padding. this sensor has binary values[0,1]. 0 means padded(null value),1 means value exists.

 


In the last notebook (sentiment_analysis_2.ipynb), I decided to use pre_trained BERT model, not as only tokenizer but also classificaion_model
