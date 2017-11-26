import json
import os
import codecs
import pandas as pd
import numpy as np
from scipy import sparse
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_files = [f for f in os.listdir("data") if f.endswith(".json")]


NEGATIVE_REVIEW_MAX = 2
POSITIVE_REVIEW_MIN = 4

def review_sentiment(score):
    if score <= NEGATIVE_REVIEW_MAX:
        return 0
    elif score >= POSITIVE_REVIEW_MIN:
        return 1
    return -1

def clean_review_text(rev):
    """
        Lemmatize and lowercase everything, also remove punctuation
    """
    rev = rev.lower()
    tokens = rev.split()
    lemmatized_tokens = []
    for t in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(t.strip(".,!'")))
        
    return ' '.join(lemmatized_tokens)


SPLIT_SIZE = 200000
for f in data_files:
	datas = []
	if not os.path.isfile("data/"+f.replace(".json", ".csv")):
		with codecs.open("data/"+f) as fp:
			for l in fp:
				rev = json.loads(l)
				datas.append(rev)

			data = pd.DataFrame.from_records(datas)
			data["sentiment"] = data["overall"].map(review_sentiment)
			# remove middling reviews
			data = data[data.sentiment != -1]
			data["clean_text"] = data.reviewText.map(clean_review_text)

			# only keep columns we want
			data = data[["clean_text", "sentiment"]]
			file = "data/" + f.replace(".json", "") + ".csv"


			if data.shape[0] > 2*SPLIT_SIZE:
				print("splitting file", f)
				for d in range(int(data.shape[0]/SPLIT_SIZE)):
					split_frame = data[d*SPLIT_SIZE:(d+1)*SPLIT_SIZE]
					file = "data/" + f.replace(".json", "") + "_" + str(d) +".csv" 
					print("saved", file)
					split_frame.to_csv(file, index=False)
			else:
				data.to_csv(file, index=False)

			print("done with", f)
			print(data.shape)
	else:
		print("skipping", f)