# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:42:54 2020

@author: Rocky
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
# Set the output directory for saving model file
# Optionally, set a GCP bucket location
#
#OUTPUT_DIR = 'OUTPUT_DIR_NAME'#@param {type:"string"}
##@markdown Whether or not to clear/delete the directory and create a new one
#DO_DELETE = False #@param {type:"boolean"}
##@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
#USE_BUCKET = True #@param {type:"boolean"}
#BUCKET = 'BUCKET_NAME' #@param {type:"string"}
#
#if USE_BUCKET:
#  OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)
#  from google.colab import auth
#  auth.authenticate_user()
#
#if DO_DELETE:
#  try:
#    tf.gfile.DeleteRecursively(OUTPUT_DIR)
#  except:
#    # Doesn't matter if the directory didn't exist
#    pass
#tf.gfile.MakeDirs(OUTPUT_DIR)
#print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
#


from tensorflow import keras
import os
import re

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
#  dataset = tf.keras.utils.get_file(
#      fname="aclImdb.tar.gz", 
#      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
#      extract=True)
  dataset = 'D:\\imdb_data\\'
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  
  return train_df, test_df



train, test = download_and_load_datasets()


train = train.sample(5000)
test = test.sample(5000)
train.columns
DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'polarity'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1]


# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()
tokenizer.tokenize("This here's an exam/ple of using the BERT tokenizer")
# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


import tensorflow_datasets as tfds
ds = tfds.load('imdb_reviews', data_dir = 'D:\\imdb_data\aclImdb')









