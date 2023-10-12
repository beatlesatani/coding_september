# sentiment_analysis
this is a repository of sentiment_analysis for the September coding challenge.
My sentiment analysis web application takes a sentence as an input and returns the predicted sentiment from "negative, neutral, positive."
This sentiment analysis project consists of two parts.
1. classifier modeling part
1. web application part

In this sentiment_analysis, I used the dataset taken from twitter,train.csv,test.csv.
※As a practice of NLP, I worked on the text_classification model in the NLP_practice notebook.

## 1. Classifier modeling
 In the model, I adopted the BERT model as a word embedding, not word2vec. An embedding is a compressed representation of data, such as text or images, as continuous vectors in a lower-dimensional space so that the ML/DL model can take sentences as inputs. BERT's bidirectional encoding strategy allows it to ingest the position of each word in a sequence and incorporate that into that word's embedding. In contrast, Word2Vec embeddings aren't able to account for word position. In other words, BERT models can take context into the model (ex. Minute. This word has two meanings. 1. the unit of time. 2. small. Usual word embedding methods like word2vec assign the same vector to two different 'minute.' However, the BERT model assigned other vectors to two-different 'minute').
Here, I made three notebooks for making the sentiment_analysis model,

In the first notebook (sentiment_analysis_1.ipynb), I made a sentiment analysis model based on the lecture video on YouTube. I applied a pre-trained Bert model(preprocessor, encoder) in this model. After that, I put embedding inputs into superficial Neural Network Layers and trained the model with ten epochs. However, the accuracy was lower than 60%, much lower than expected. This score required superficial neural network layers to make good predictions. In addition, there needed to be more suitable than just applying a pre-trained Bert encoder.

In the second notebook (sentiment_analysis_2.ipynb), I used the transformer library from Hugging-face. Hugging Face Transformers is an open-source framework for deep learning created by Hugging Face. It provides APIs and tools to download state-of-the-art pre-trained models and tune them further. I used a pre_train tokenizer named bert-base-cased from this library to set up embedding. In general, the Classification model takes two inputs, input_id and attention_mask. 


- 1.Input_ids
    input_ids are simply the numeric representations of the tokens. Single words are converted into numerical values. In addition, there are special tokens exist,[SEP],[CLS],[PAD]. [SEP](token = 102) is added at the end of the sentence. [CLS] (token = 101)is added at the beginning of the sentence. [PAD](token = 0) means padding.

  2.Attention_mask
  Attention_mask is useful when we add padding to the input tokens. The attention mask tells us which input_ids correspond to padding. this sensor has binary values[0,1]. 0 means padded(null value),1 means value exists.

After setting up the encoder, I made a classifier from scratch. However, it took too long time to train the model, so I gave up on making the original classifier.
 


In the last notebook (sentiment_analysis_submit.ipynb), I decided to use a pre-trained BERT model as a tokenizer and a classification model. The data pre-processing method is the same as the second notebook. In the building model part, I retrieved the pre-trained classification model named TFBertForSequenceClassification from the huggingface. This classifier also takes input_ids and attention_masks as inputs. This model makes outcomes, called logits, the unnormalized scores or predictions. These logits represent the model's raw predictions for each class or category in your classification task. To get normalized probability, I applied the softmax function. Through the training process, the accuracy of this model is 93.41%. I saved and exported this model into Google Drive for a web application.

## 2.Web Application 
in the web application process, I used Flask ngrok. Flask enables web application development on Google Colab and runs on a local host over the internet with the ngrok tool. This web application takes input sentences from the user and predicts the sentiment of each sentence.

## Future Development
As a future development, there are two parts which will be improved. 

- 1. Model Accuracy
     Removing stop words is helpful. Stop words are used often but mean nothing (I am, a, an, the).
- 2. Web Application
     With Apache Spark, it can build a real-time sentiment prediction model.

