import gensim
import json
import codecs
import pandas as pd
import numpy as np
import re
import os

not_in_vocab = set()
NEGATIVE_REVIEW_MAX = 2
POSITIVE_REVIEW_MIN = 4

def get_embedding(rev):
	# first, map phrases in word embedding into their reviews
	for p in phrases:
		if " "+p.replace("_", " ")+" " in rev:
			rev = rev.replace(p.replace("_", " "), p)

	vec = None
		# now split the review 
	rev_sp = rev.split()
	for r in rev_sp:
		if r in model.wv.vocab:
			if vec is None:
				vec = model.wv[r]
			else:
				vec = np.vstack((vec, model.wv[r]))
		else:
			not_in_vocab.add(r)
	if vec is None:
		vec = np.zeros(300)
	return np.mean(vec, axis=0)

def review_sentiment(score):
	if score <= NEGATIVE_REVIEW_MAX:
		return 0
	elif score >= POSITIVE_REVIEW_MIN:
		return 1
	return -1


model = gensim.models.KeyedVectors.load_word2vec_format("w2v/GoogleNews-vectors-negative300.bin", binary=True)

phrases = sorted([p for p in model.wv.vocab if re.search("\w_\w", p)], key=lambda x:len(x), reverse=True)

phrase_dict = {p:1 for p in phrases}


data_files = sorted([f for f in os.listdir("data") if f.endswith(".json")], key=lambda x:os.path.getsize("data/" +x))


SPLIT_SIZE = 200000
for f in data_files:
	datas = []
	if not os.path.isfile("data/"+f.replace(".json", "_word_vecs.csv")):
		print("gettings vecs for ", f)
		with codecs.open("data/"+f, encoding='utf-8') as fp:
			for l in fp:
				r = json.loads(l)
				datas.append(r)

		data = pd.DataFrame.from_records(datas)
		data["sentiment"] = data["overall"].map(review_sentiment)
		data["vec"] = data["reviewText"].map(get_embedding)
		print("done getting vectors")
		data = data[["vec", "sentiment"]]
		data.to_csv(file.replace(".json", "_word_vecs.csv"))
		print("saved file, ", f)
	else:
		print("skipping", f)