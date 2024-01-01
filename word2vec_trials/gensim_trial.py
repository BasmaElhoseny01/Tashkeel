from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize  # or any other tokenizer of your choice

# Example tokenization using NLTK
text_data = "Your text data goes here. It can be a single string or a list of strings."
tokenized_data = [word_tokenize(sentence) for sentence in text_data.split('.')]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Save model
model.save("word2vec_model.bin")


# Access word vectors
# vector = model.wv['example_word']
similar_words = model.wv.most_similar('text', topn=5)
print(similar_words)
# # Load model
# loaded_model = Word2Vec.load("word2vec_model.bin")
