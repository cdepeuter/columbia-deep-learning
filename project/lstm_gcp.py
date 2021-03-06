import tensorflow as tf
import numpy as np
import os
import re
import datetime
import codecs
import sys

current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M")
TEST_SIZE= 2084
iterations = 1000
maxSeqLength = 50
numDimensions = 300
numClasses = 2
BATCH_SIZE = 2048
lstmUnits = 16

if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
    lstmUnits = int(sys.argv[2])
    iterations = int(sys.argv[3])

print("batch size", BATCH_SIZE)
print("lstmUnits", lstmUnits)
print("iterations", iterations)


with codecs.open("logs/run_"+current_time+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
    fp.write("lstmUnits " + str(lstmUnits) + "\n")
    fp.write("iterations " + str(iterations) + "\n")


# load the GLOVE arrays, smaller than w2v google, hopefully no perfomance drop
# VECTORS_FILE = 'data/wordVectors.npy'
# DATA_FILE_START = "reviews"


# # google news top 200,000 words

# VECTORS_FILE = "data/w2v_vectors.npy"
# DATA_FILE_START = "w2vreviews"

VECTORS_FILE = "data/vecs/w2v_vectors.npy"
DATA_FILE_START = "balanced_w2vreviews"

wordVectors = np.load(VECTORS_FILE)
print("wordVectors shape", wordVectors.shape)



def getTrainData():
    train_files = [f for f in os.listdir("data/vecs") if f.startswith(DATA_FILE_START) and f.endswith(".npy")]
    
    frames = [np.load("data/vecs/" + f) for f in train_files]
    labels = [np.load("data/vecs/" + f.replace("reviews", "labels")) for f in train_files]
    
    X = np.vstack(frames)
    y = np.vstack(labels)
    
    return X.astype(int),y.astype(int)

train_data, train_labels = getTrainData()


MAX_TRAIN_SIZE = 3000000
train_data = train_data[0:MAX_TRAIN_SIZE]
train_labels = train_labels[0:MAX_TRAIN_SIZE]

print("train data shape", train_data.shape)
print("train labels shape", train_labels.shape)
print("train data max:", train_data.max())
print("train data balance", train_labels.mean(axis=0))



def getTrainBatch(size=None):
    global train_data
    global train_labels
        
  
    if size is not None:
        #ix = np.array(range(size))
        ix = np.random.randint(train_data.shape[0], size=size)
    


    return train_data[ix,], train_labels[ix, ]

# def getYelpData():
#     arr = np.load("data/vecs/_test_yelp_w2vreviews_0.npy")
#     labels = np.load("data/vecs/_test_yelp_w2vlabels_0.npy")
#     return arr, labels


def getTestBatch(size=None):
    

    arr = np.load("data/vecs/"+DATA_FILE_START.replace("balanced", "test")+"_0.npy")
    labels = np.load("data/vecs/"+DATA_FILE_START.replace("balanced", "test").replace("reviews", "labels")+"_0.npy")
    
    if size is not None:
        
        ix = np.random.randint(arr.shape[0], size=size)
        arr = arr[ix,]
        labels = labels[ix,]
    
    return arr, labels


def one_hot_label(label):
    if label==0:
        return np.array([1,0])
    else:
        return np.array([0,1])


# yelp_data, yelp_labels = getYelpData()

# print("yelp data shape", yelp_data.shape)



test_data, test_labels = getTestBatch(size=TEST_SIZE)
print(len(test_labels), test_labels.shape)
print(test_data.shape)
print("test data balance", test_labels.mean(axis=0))


labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.int64, [None, maxSeqLength])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("embeds"):
    embedding = tf.get_variable(name="word_embedding", shape=wordVectors.shape, initializer=tf.constant_initializer(wordVectors), trainable=False)


#data = tf.Variable(tf.zeros([None, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(embedding,input_data)
#print("data", data[1,])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits,forget_bias=1)
# outputs,_ =tf.contrib.rnn.static_rnn(lstmCell,input,dtype="float32")

with tf.variable_scope("lstm"):

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    #outputs, states = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    value, _ = tf.nn.dynamic_rnn(lstmDropout, data, dtype=tf.float32)

with tf.variable_scope("final_layer"):

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), name="last_layer_weights")
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name="last_layer_biases")
    value = tf.transpose(value,   [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()


sess = tf.InteractiveSession()
logdir = "tensorboard/" + current_time + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# dont save last layer weights
all_vars = tf.all_variables()
var_to_restore = [v for v in all_vars if v.name.split('/')[0]=='embed' or v.name.split('/')[0]=='lstm']
var_names = [v.name for v in var_to_restore]
saver = tf.train.Saver(var_to_restore)


print("variables to restore", var_names)

sess.run(tf.global_variables_initializer())

final_yelp_acc = None
final_test_acc = None



for i in range(iterations):
    #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch(size=BATCH_SIZE)
    _, summary, acc, pred, loss_, outputs = sess.run([ train_step, merged, accuracy, prediction, loss, value], {input_data: nextBatch, labels: nextBatchLabels,keep_prob:.75})
    #Save the network every 10,000 training iterations
    if i % 10 == 0:
        
        writer.add_summary(summary, i)
        print("\n")
        print("step %d" % i)
        print("train accuracy %f" % acc)
        print("loss: %f" % loss_)
        print("balance %f, %f" % (nextBatchLabels.mean(axis=0)[0], nextBatchLabels.mean(axis=0)[0] + acc))
        #print("pred mean %f" % np.mean(pred))

    if i % 50 == 0 or i == iterations-1:
        final_test_acc = accuracy.eval({input_data:test_data, labels: test_labels, keep_prob:1.0})
        # final_yelp_acc = accuracy.eval({input_data:yelp_data, labels: yelp_labels, keep_prob:1.0})
        print("****************")
        print("test accuracy:  % f" % final_test_acc)
        # print("yelp accuracy:  % f" % final_yelp_acc)
        print("\n\n")
#         print("mean prediction", np.mean(pred))
#         print("\n\n")
        


with codecs.open("logs/run_final_"+current_time+".txt",'w') as fp:
    fp.write("time " + current_time + "\n")
    fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
    fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
    fp.write("lstmUnits " + str(lstmUnits) + "\n")
    fp.write("iterations " + str(iterations) + "\n")
    fp.write("test accuracy " + str(final_test_acc) + "\n")
    # fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")


save_file_path = "./models/final_lstm"+current_time+".ckpt"
save_path = saver.save(sess, save_file_path)
print("saved to %s" % save_file_path)
writer.close()

