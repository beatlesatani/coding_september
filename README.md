# sentiment_analysis
this is a repository of sentiment_analysis for September coding challenge.
My sentiment analysis web application takes a sentence as an input, and return the predicted sentiment from "negative,neutral,positive".
This sentiment analysis project consists of two parts.
1. modeling part
2. web application part

※As a practice of NLP, I worked on the text_classification model in the NLP_practice notebook.

## 1.modeling part
In the model, I adopted BERT model as a word embedding, not word2vec. An embedding is a compressed representation of data such as text or images as continuous vectors in a lower-dimensional space so that ML/DL model can take sentences as inputs. BERT’s bidirectional encoding strategy allows it to ingest the position of a each word in a sequence and incorporate that into that word’s embedding, while Word2Vec embeddings aren’t able to account for word position. In other words, BERT models is able to takes context into the model (ex. minute. this word has 2 meanings. 1.the unit of time.2.small. Usual word embedding methods like word2vec assigned same vector to two-different 'minute'. However in BERT model, it assigned different vectors to two-different 'muinute').
Here,I made three notebooks for making sentiment_analysis model,


