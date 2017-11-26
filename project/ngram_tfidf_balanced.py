import json
import os
import codecs
import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
from google.cloud import storage
import io
import amazon_data_files
from sklearn.utils import shuffle


SEED = 21
client = storage.Client()
bucket = client.get_bucket('columbia-deeplearning-project-cdp')
data_frames = []
for b in amazon_data_files.bucket_data_files:
        blob = bucket.get_blob(b)
        dat_str = blob.download_as_string()
        frame = pd.read_csv(io.StringIO(unicode(dat_str)))
        print("frame loaded", b, frame.shape)
        data_frames.append(frame)


data = pd.concat(data_frames)
del data_frames
print("data loaded", data.shape)
print(data.shape)
data = data[data.clean_text.notnull()]


# balance subclasses
bad = shuffle(data[data.sentiment == 0], random_state=SEED)
good = shuffle(data[data.sentiment == 1], random_state=SEED)
data = pd.concat([bad, good[0:bad.shape[0]]])
del bad
del good


print("mean sentiment", data.sentiment.mean())
print("data shape", data.shape)
print(data.columns)
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
ngram_vectorizer = HashingVectorizer(stop_words='english', n_features=500000, ngram_range=(1,5))
bag_o_ngram = ngram_vectorizer.fit_transform(data["clean_text"])
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
tfidf_vec = tfidf_vectorizer.fit_transform(data["clean_text"])
tfidf_ngram = sparse.hstack((tfidf_vec, bag_o_ngram))


from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(tfidf_ngram,  data.sentiment, test_size=1.0/6, random_state=42)

print(tfidf_ngram.shape)
print(train_data.shape)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(train_data, train_labels)
print("tfidf ngram  model fit")
preds = lr_model.predict(test_data)

accuracy = np.mean(preds == test_labels.values)
print("tfidf_ngram accuracy", accuracy)


# write results to file
cur_time = str(datetime.now())
outfile_name = "results/results_ngram_tfidf_balanced" + cur_time.replace(" ", "_") + ".txt"
with codecs.open(outfile_name, 'w') as fp:
        fp.write("balanced ngram tfidf accuracy: " + str(accuracy) +  "\n")