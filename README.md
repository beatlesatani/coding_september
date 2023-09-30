# sentiment_analysis
this is a repository of sentiment_analysis for the September coding challenge.
My sentiment analysis web application takes a sentence as an input and returns the predicted sentiment from "negative, neutral, positive".
This sentiment analysis project consists of two parts.
1. classifier modeling part
1. web application part

In this sentiment_analysis, I used the dataset, which was taken from twitter,train.csv,test.csv.
※As a practice of NLP, I worked on the text_classification model in the NLP_practice notebook.

## 1. Classifier modeling
 In the model, I adopted the BERT model as a word embedding, not word2vec. An embedding is a compressed representation of data, such as text or images, as continuous vectors in a lower-dimensional space so that the ML/DL model can take sentences as inputs. BERT’s bidirectional encoding strategy allows it to ingest the position of each word in a sequence and incorporate that into that word’s embedding, while Word2Vec embeddings aren’t able to account for word position. In other words, BERT models are able to take context into the model (ex. minute. this word has 2 meanings. 1. the unit of time. 2. small. Usual word embedding methods like word2vec assign the same vector to two different 'minute'. However, in the BERT model, it assigned different vectors to two-different 'minute').
Here, I made three notebooks for making the sentiment_analysis model,

In the first notebook (sentiment_analysis_1.ipynb), I made a sentiment analysis model based on the lecture video on YouTube. In this model, I applied a pre-trained bert model(preprocessor, encoder). After that, I put embedding inputs into simple Neural Network Layers, and I trained the model with 10 epochs. However, the accuracy was lower than 60%, which  was much lower than I expected. As a reason for this score, simple neural network layers were not enough to make good predictions. In addition, just applying a pre-trained Bert encoder was not suitable.

In the second notebook (sentiment_analysis_2.ipynb), I decided to use the transformer library from Hugging-face. Hugging Face Transformers is an open-source framework for deep learning created by Hugging Face. It provides APIs and tools to download state-of-the-art pre-trained models and further tune them. From this library, I used a pre_train tokenizer named bert-base-cased to set up embedding. In general, the Classification model takes two inputs, input_id and attention_mask. 


- 1.Input_ids
    input_ids are simply the numeric representations of the tokens. Simply single words are converted into numerical value. In addition, there are special tokens exist,[SEP],[CLS],[PAD]. [SEP](token = 102) is added at the end of the sentence. [CLS] (token = 101)is added at the beginning of the sentence. [PAD](token = 0) means padding.

- 1. attention_mask
  Attention_mask is useful when we add padding to the input tokens. The attention mask tells us which input_ids correspond to padding. this sensor has binary values[0,1]. 0 means padded(null value),1 means value exists.

After setting up the encoder, I worked on making a classifier from scratch. However, it took too long time to train the model, so I gave up on making the original classifier.
 


In the last notebook (sentiment_analysis_2.ipynb), I decided to use a pre-trained BERT model, not only as a tokenizer but also as a classification model. The data pre-processing method is exactly the same as the second notebook. In the building model part, I retrieved the pre-trained classification model named TFBertForSequenceClassification from the huggingface. This classifier also takes input_ids and attention_masks as inputs. This model makes outcome, which is called logits, the unnormalized scores or predictions. These logits represent the model's raw predictions for each class or category in your classification task. To get normalized probability, I applied the softmax function. Through the training process, the accuracy of this model is ????????????%. In the end, I saved and exported this model into Google Drive so that I could use it in a web application.

## 2.Web Application 
   - in the web application process, I used Flask ngrok. Flask enables to development of the web application on google colab and run on a local host over the internet with ngrok tool. This web application takes input sentences from the user and predicts the sentiment of each sentence.



https://docs.github.com/ja/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
youtubeで動くところをリンク限定で見せる   
