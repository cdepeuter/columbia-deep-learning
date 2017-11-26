# Sentiment Analysis using LSTM

In this project we are investigating the effectiveness of LSTM for sentiment analysis tasks. We are using the amazon review data (http://jmcauley.ucsd.edu/data/amazon/), mapping reviews of 1 and 2 stars to a 0 sentiment, and 4 and 5 stars to a 1 sentiment, following the procedure in Zhang et al (https://arxiv.org/pdf/1509.01626.pdf).

### Data Cleaning

We first clean the reviews in clean_data.py. Cleaning involves lemmatization (not done in the paper), lower casing all words, and the sentiment score to binary mapping.  

### Baseline models:

We first build 4 different baseline models as in the paper:
	- Bag of words: (take the top 50,000 features)
	- Bag of words + tfidf 
	- Ngrams (top 500,000 1-5 grams)
	- Ngrams + tfidf

We use reviews across a wide range of products (see amazon_data_files.py), and train on 3,000,000 reviews, and test on 600,000 reviews. The models are run and results are written to the /results folder using Google Cloud Platform.


### LSTM:

We also implement a basic LSTM following the procedure of the paper, using the review mean word-embeddings as the features for each review (300 dimensional)

### Looking at class imbalance:
For the baselines, we also observe how the models do when the classes are balanced. We believe this is an oversight of the original paper, as we found the data to have a 9:1 ratio between positive and negative reviews. We were interested to see how the baselines would perform. Results for the balanced baseline models are also written to the /results folder.


