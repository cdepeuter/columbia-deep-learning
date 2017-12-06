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
numClasses = 5
BATCH_SIZE = 2048
lstmUnits = 16

if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
    lstmUnits = int(sys.argv[2])
    iterations = int(sys.argv[3])

print("batch size", BATCH_SIZE)
print("lstmUnits", lstmUnits)
print("iterations", iterations)


with codecs.open("logs/retrained_"+current_time+".txt",'w') as fp:
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
DATA_FILE_START = "yelp_w2vreviews"

wordVectors = np.load(VECTORS_FILE)
print("wordVectors shape", wordVectors.shape)



def getYelpData():
    train_files = [f for f in os.listdir("data/vecs") if f.startswith(DATA_FILE_START) and f.endswith(".npy")]
    
    frames = [np.load("data/vecs/" + f) for f in train_files]
    labels = [np.load("data/vecs/" + f.replace("reviews", "ratings")) for f in train_files]
    
    X = np.vstack(frames)
    y = np.vstack(labels)
    
    return X.astype(int),y.astype(int)



def getTrainBatch(size=None):
    global train_data
    global train_labels
        
  
    if size is not None:
        #ix = np.array(range(size))
        ix = np.random.randint(train_data.shape[0], size=size)
    
    return train_data[ix,], train_labels[ix, ]

def getYelpTest():
    arr = np.load("data/vecs/test_yelp_w2vreviews_0.npy")
    labels = np.load("data/vecs/test_yelp_w2vratings_0.npy")
    return arr, labels


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




train_data, train_labels = getYelpData()

print("train data shape", train_data.shape)
print("train labels shape", train_labels.shape)
print("train data max:", train_data.max())
print("train data balance", train_labels.mean(axis=0))


test_data, test_labels = getYelpTest()
print(len(test_labels), test_labels.shape)
print(test_data.shape)
print("test data balance", test_labels.mean(axis=0))


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.int64, [None, maxSeqLength])

keep_prob = tf.placeholder(tf.float32)
embedding = tf.get_variable(name="word_embedding", shape=wordVectors.shape, initializer=tf.constant_initializer(wordVectors), trainable=False)


#data = tf.Variable(tf.zeros([None, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(embedding,input_data)
#print("data", data[1,])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits,forget_bias=1)
# outputs,_ =tf.contrib.rnn.static_rnn(lstmCell,input,dtype="float32")
with tf.variable_scope("lstm"):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    #outputs, states = tf.nn.dxqynamic_rnn(lstmCell, data, dtype=tf.float32)

    value, _ = tf.nn.dynamic_rnn(lstmDropout, data, dtype=tf.float32)

with tf.variable_scope("final_layer"):
    # old weights will load, but we dont care about that
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value,   [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

all_vars = tf.all_variables()
print("all variables", [v.name for v in all_vars])

var_names = [v for v in all_vars if v.name.split("/")[0] == 'final_layer']
print("variables to train")
print(var_names)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss, var_list=var_names)


print("NAMES", var_names)

load_path = "./models/final_lstm20171205-1938.ckpt"
restore_vars = [v for v in all_vars if v.name.split("/")[0] == "lstm"]
sess = tf.InteractiveSession()
saver = tf.train.Saver(var_list=restore_vars)
# Initialize v1 since the saver will not.
saver.restore(sess, load_path)


tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + current_time + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())
final_yelp_acc = None
final_test_acc = None




final_test_acc = accuracy.eval({input_data:test_data, labels: test_labels, keep_prob:1.0})
print("****************")
print("test accuracy:  % f" % final_test_acc)

for i in range(iterations):
    #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch(size=BATCH_SIZE)
    _, summary, acc, loss_,  = sess.run([ train_step, merged, accuracy, loss], {input_data: nextBatch, labels: nextBatchLabels,keep_prob:.75})
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
        print("****************")
        print("test accuracy:  % f" % final_test_acc)
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
    fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")

        


# with codecs.open("logs/run_final_"+current_time+".txt",'w') as fp:
#     fp.write("time " + current_time + "\n")
#     fp.write("BATCH_SIZE " + str(BATCH_SIZE) + "\n")
#     fp.write("TEST SIZE " + str(TEST_SIZE) + "\n")
#     fp.write("lstmUnits " + str(lstmUnits) + "\n")
#     fp.write("iterations " + str(iterations) + "\n")
#     fp.write("test accuracy " + str(final_test_acc) + "\n")
#     fp.write("yelp accuracy " + str(final_yelp_acc) + "\n")
