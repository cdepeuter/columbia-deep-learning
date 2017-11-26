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


client = storage.Client()
bucket = client.get_bucket('columbia-deeplearning-project-cdp')
data_frames = []
for b in amazon_data_files.bucket_data_files:
        blob = bucket.get_blob(b)
        dat_str = blob.download_as_string()
        frame = pd.read_csv(io.StringIO(unicode(dat_str)))
        print("frame loaded", b)
        print(frame.shape)
        data_frames.append(frame)

data = pd.concat(data_frames)
del data_frames
print("data loaded")
print(data.shape)

data = data[data.clean_text.notnull()]
if data.shape[0]> 3600000:
        data = data.sample(n=3600000, random_state=42)
print(data.shape)
print(data.columns)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=50000)
bow = vectorizer.fit_transform(data["clean_text"])

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(bow,  data.sentiment, test_size=1.0/6, random_state=42)
print(bow.shape)
print(train_data.shape)


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(train_data, train_labels)
print("lr bow  model fit")
preds = lr_model.predict(test_data)
accuracy = np.mean(preds == test_labels.values)
print("bow accuracy", accuracy)

cur_time = str(datetime.now())
# write results to file
outfile_name = "results/results_bow_" + cur_time.replace(" ", "_") + ".txt"
with codecs.open(outfile_name, 'w') as fp:
        fp.write("bag of words accuracy: " + str(accuracy) + "\n")
