harakat={
  "Fatha":"\u064e", #Fatha
  "Fathatan":  "\u064b", #Fathatan
  "Damma":"\u064f", #Damma
  "Dammatan":"\u064c" , #Dammatan
  "Kasra":"\u0650" ,  #Kasra
  "Kasratan":"\u064d" ,  #Kasratan
  "Sukun":"\u0652", #Sukun
  "Shadda":"\u0651", #Shadda

  "Shadda Fatha":"\u0651\u064e",  #Shadda Fatha
  "Shadda Fathatan":"\u0651\u064b", #Shadda Fathatan
  "Shadda Damma":"\u0651\u064f", #Shadda Damma
  "Shadda Dammatan":"\u0651\u064c", #Shadda Dammatan
  "Shadda Kasra":"\u0651\u0650", #Shadda Kasra
  "Shadda Kasratan":"\u0651\u064d", #Shadda Kasratan
}
classes={
    harakat["Fatha"]: 0, #Fatha
      harakat["Fathatan"]: 1, #Fathatan
      harakat["Damma"]: 2, #Damma
      harakat["Dammatan"]: 3, #Dammatan
      harakat["Kasra"]: 4,  #Kasra
      harakat["Kasratan"]: 5,  #Kasratan
      harakat["Sukun"]: 6, #Sukun
      harakat["Shadda"]: 7, #Shadda
      
      harakat["Shadda Fatha"]: 8,  #Shadda Fatha
      harakat["Shadda Fathatan"]: 9, #Shadda Fathatan
      harakat["Shadda Damma"]: 10, #Shadda Damma
      harakat["Shadda Dammatan"]: 11, #Shadda Dammatan
      harakat["Shadda Kasra"]: 12, #Shadda Kasra
      harakat["Shadda Kasratan"]: 13, #Shadda Kasratan
      "": 14 #Undiacrtized
}
n_classes=len(classes.keys())

import pickle
from utils import one_hot_encode
vocab=[]
with open('/content/drive/MyDrive/Tashkeel/arabic_letters.pickle', 'rb') as file:
    vocab = list(pickle.load(file))


    # Add <s>
    vocab.append('<s>')
    # Add </s>
    vocab.append('</s>')

    # Add space
    vocab.append(' ')

    # Add pad
    vocab.append('<PAD>')
    
    # Add <UNK>
    vocab.append('<UNK>')

    vocab_map = {char: index for index, char in enumerate(vocab)}
    print("Loaded Vocab",len(vocab_map.keys()),vocab_map)

    
    # Get the size of the vocabulary
    vocab_size = len(vocab)

    # Getting one hot encoded ditionary
    # Create a mapping from each unique element to its index
    # vocab_mapping = {element: index for index, element in enumerate(vocab)}


    # id2char = {i: word for i, word in enumerate(list(vocab))}
    # char2id = {word: i for i, word in id2char.items()}

    
    # Create a dictionary mapping each element to its one-hot encoded representation
    # one_hot_encoding = {element: one_hot_encode(index, vocab_size) for element, index in vocab_mapping.items()}
    # print("one_hot_encoding is avalaible",one_hot_encoding)






