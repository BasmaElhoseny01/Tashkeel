import numpy as np

from itertools import zip_longest

import torch
from torch import nn

# Defines a custom dataset class for PyTorch (used for handling data)
from torch.utils.data import Dataset

# Creates a DataLoader for efficient batch processing in PyTorch (used for data loading)
from torch.utils.data import DataLoader

# Splits a dataset into training and validation sets (used for data splitting)
from torch.utils.data import random_split

import pickle

import string
import re
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import nltk
import numpy as np
import pickle
nltk.download('punkt')
from tqdm import tqdm

#  Cross Entropy Loss function (used for Multi classification problems)
from torch.nn import CrossEntropyLoss



def get_params(vocab, tashkeel_map, sentences_file, tashkeel_file):
    print(vocab)
    print(tashkeel_map)
    sentences = []
    labels = []

    # Convert Char to its id
    with open(sentences_file, 'rb') as file:
          X = pickle.load(file)  #[['ا']....]
          # Replace Char by its id in the vocab
          for sentence in X:
            s = [vocab[char] if char in vocab 
                 else vocab['<UNK>']
                 for char in sentence]
            sentences.append(s)


    with open(tashkeel_file, 'rb') as file:
          y = pickle.load(file)  #[['ا']....]
          labels=y
          
    return sentences, labels, len(sentences)


# Function to generate one-hot encoding for an element
def one_hot_encode(element_index, size):
    encoding = np.zeros(size)
    encoding[element_index] = 1
    return encoding
    
# def char_to_one_hot(char, vocab):
#     one_hot = np.zeros(len(vocab))
#     if char in vocab:
#         index = vocab.index(char)
#         one_hot[index] = 1
#     else:
#       # Unknown
#       index = vocab.index('<UNK>')
#       one_hot[index] = 1
#     return one_hot
  