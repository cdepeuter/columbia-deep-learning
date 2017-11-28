# Sentiment Analysis using LSTM

In this project we are investigating the effectiveness of LSTM for sentiment analysis tasks. We are using the amazon review data (http://jmcauley.ucsd.edu/data/amazon/), mapping reviews of 1 and 2 stars to a 0 sentiment, and 4 and 5 stars to a 1 sentiment, following the procedure in Zhang et al (https://arxiv.org/pdf/1509.01626.pdf).

### Data Cleaning

We first clean the reviews using https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/clean_data.py. Cleaning involves lemmatization (not done in the paper), lower casing all words, and the sentiment score to binary mapping. All the cleaned reviews are put in a Google Cloud Platfrom storage bucket. Cleaning is done simply running the command 

`$ python clean_data.py` 

This command looks for all .json files in the /data directory (not uploaded to repo but can be downloaded at the link above), does all of the processing mentioned above, and saves a .csv of the cleaned data to the same directory. 

### Baseline models:

We first build 4 different baseline models as in the paper:
* Bag of words: (take the top 50,000 features)
    - `$ python bow.py `
* (https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/bow.py)
* Bag of words + tfidf (https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/bow_tfidf.py)
    - `$ python bow_tfidf.py`
* Ngrams (top 500,000 1-5 grams)
    - `$ python ngram.py`
* (https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/ngrams.py)
* Ngrams + tfidf (https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/tfidf_ngram.py)
    - `$ python tfidf_ngram.py `

We use reviews across a wide range of products (https://github.com/cdepeuter/columbia-deep-learning/blob/master/project/amazon_data_files.py), and train on 3,000,000 reviews, and test on 600,000 reviews. The models are run and results are written to the /results folder using Google Cloud Platform. When each model is run i.e. `$ python bow.py` the reviews are pulled from the bucket in a deterministic order, and read into a dataframe. The features are calcualted and then the data is split into train/test sets. The baseline models are all Logistic Regression as in the paper.


### LSTM:

We also implement a basic LSTM following the procedure of the paper, using the review mean word-embeddings as the features for each review (300 dimensional). The model is implement in this jupyter notebook: https://github.com/cdepeuter/columbia-deep-learning/blob/master/LSTM_deep_learning_reviews.ipynb
This notebook assumes the data files are in a /project folder, and assumes the standard GLOVE vectors have been downloaded and placed in that folder as well. Our LSTM implementation draws heavily from the tutorial here: https://github.com/adeshpande3/LSTM-Sentiment-Analysis

### Looking at class imbalance:
For the baselines, we also observe how the models do when the classes are balanced. We believe this is an oversight of the original paper, as we found the data to have a 9:1 ratio between positive and negative reviews. We were interested to see how the baselines would perform. Results for the balanced baseline models are also written to the /results folder.


