####################################################################
#Author: Thanh Tran                                                #
#Cite Sources:                                                     #
#   Andy Thomas: http://adventuresinmachinelearning.com            #
####################################################################
import numpy as np
import pandas as pd
import collections, string, re, random, math
import datetime as dt
from nltk.tokenize import RegexpTokenizer

def tokenize(sentence):
    """
        input: a sentence
        return: tokens in the sentence
    """
    # tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    tokens = sentence.replace(",", " ").replace(",", ".").replace('.', ' . ').strip().split()
    return tokens

def preprocessing_corpus(corpus_path, n_words):
    """
        input: cmpsc courses catalog copy from the web
        process: extract the most common words (count them up), assign one hot location, reversed one hot look up
        return:
            tokens_location: list of tokens' associating one hot index
            one_hot_assignment: dictionary associating (words,count) with one hot index
            reversed_one_hot_assignment: dictionary associating one hot index with the (word, count)
            count: frequency of the vocabs
    """
    tokenizer = RegexpTokenizer(r'\w+')

    #parse files into sentences
    with open(fname) as f:
        corpus = f.readlines()
    corpus = [x.strip() for x in content]

    #parse corpus into tokens
    tokens = []
    for i in range(len(corpus)):
        tokens.extend(tokenize(corpus[i]))

    print(tokens)

    #process the tokens and associating the word with it's corresponding location in one-hot vector
    count = [('<UNK>', -1)]
    count.extend(collections.Counter(tokens).most_common(n_words-1)) #sorted by ascending order

    #one hot assignment. The order of the word is associate with it's frequency (highest frequency has the lowest index)
    one_hot_assignment = {}
    for i in range(len(count)):
        one_hot_assignment[count[i][0]] = i

    #get the location of the one_hot_assignment count the number of unknown word
    tokens_location = []
    unknown_count = 0
    for token in tokens:
        loc = one_hot_assignment.get(token) if (token in one_hot_assignment) else 0
        unknown_count = unknown_count+1 if loc == 0 else unknown_count
        tokens_location.append(loc)
    count[0] = ('<UNK>', unknown_count)

    #reversed look up dictionary for one_hot_encoder
    reversed_one_hot_assignment = dict(zip(one_hot_assignment.values(), one_hot_assignment.keys()))
    return (tokens, tokens_location, count, one_hot_assignment, reversed_one_hot_assignment)

data_index = 0 #global to make sure the sliding window does not repeat itself
def generate_mini_batch(data, batch_size, skip_window, num_skips):
    """
        input:
            data: input sequence of data
            batch_size: number of entries per batch
            skip_window: radius of the skipping windows
            num_skips: randomly pick num_skips from the surrounding context (w/ window radius).
                       How many times to reuse an input to generate a context.
        return:
            batch: the center of the word
            context: the surrounding context of the batch
    """
    global data_index
    assert batch_size % num_skips == 0 #
    assert num_skips <= 2*skip_window # make sure there is enough surrounding context to select from

    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # diameter + 1

    #add context to data as we slide the window of the word to the right
    #a sliding window through the context in a way with span tokens
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    #keep sliding the buffer window to the right and
    #filling in the batch array and the context array
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_select = list(buffer)
        del targets_to_select[target]
        #select num_skips tokens in the surrounding context
        for j in range(num_skips):
            target = random.choice(targets_to_select)
            targets_to_select.remove(target)
            batch[i * num_skips + j] = buffer[skip_window]  # center
            context[i * num_skips + j, 0] = target  # target from context

        #slide buffer window to the right
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context

vocabulary_size = 1000
path = './data/documents/corpus.txt'
tokens, tokens_location, count, one_hot_assignment, reversed_one_hot_assignment = preprocessing_corpus(path, vocabulary_size)

# batch, context = generate_mini_batch(tokens_location, 128, 2, 4)

######################################
#Tensorflow Model
######################################
import tensorflow as tf


#validation will randomly select 16 of the top 100 words in our vocabs to assess the model as it train
validation_size = 16
validation_window = 100
validation_examples = np.random.choice(validation_window, validation_size, replace=False)
num_sampled = 64 #number of negative examples to sample

#hyperparameters for the model
batch_size = 128
embedding_size = 16 #dimesion of the embedding vector
skip_window = 6
num_skips = 8

graph = tf.Graph()

with graph.as_default():
    #placeholders to hold input words and context words which we are trying to predict
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
    validation_dataset = tf.constant(validation_examples, dtype=tf.int32)

    #Setup embedding matrix tensor (uniform initialization for better convergence)
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    #because the objective function is not strictly convex, initialization matters
    weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))
    hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

    #convert the train context to a one-hot vector format
    train_one_hot = tf.one_hot(train_context, vocabulary_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm

    #validation and compute similarity
    validation_embeddings = tf.nn.embedding_lookup(normalized_embeddings, validation_dataset)
    similarity = tf.matmul(validation_embeddings, normalized_embeddings, transpose_b=True)

    #add a global initializer
    init = tf.global_variables_initializer()

def run(graph, num_steps):
    with tf.Session(graph = graph) as session:
        init.run() #initialize all variables
        print("initialized the variables")

        average_loss = 0.0
        for step in range(num_steps):
            batch_inputs, batch_context = generate_mini_batch(tokens_location, batch_size, skip_window, num_skips)
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

            _, loss_val = session.run([optimizer, cross_entropy], feed_dict= feed_dict)
            average_loss+= loss_val

            if(step % 2000 == 0):
                if(step > 0):
                    average_loss/=2000
                print("Average loss at step ", step, ": ",average_loss)
                average_loss = 0

            #compute on the validation set to see how things are going
            if(step % 10000 == 0):
                sim = similarity.eval()
                for i in range(validation_size):
                    validation_words = reversed_one_hot_assignment[validation_examples[i]]
                    top_k = 8 #select the nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" % validation_words

                    for k in range(top_k):
                        close_word = reversed_one_hot_assignment[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
        save_path = saver.save(session, "../models/skipgrams/model.ckpt")
        print("Model saved in path: %s" % save_path)
# num_steps = 100
# softmax_start_time = dt.datetime.now()
# run(graph, num_steps=num_steps)
# softmax_end_time = dt.datetime.now()
# print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))

with graph.as_default():

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 200000
nce_start_time = dt.datetime.now()
run(graph, num_steps)
nce_end_time = dt.datetime.now()
print("NCE method took {} minutes to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))












#
