from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import time


# Download NLTK resources if not already downloaded
nltk.download('punkt')

start_time = time.time()
# Load Arabic text from a file
print(str(0) + ': Loading text ...')
file_path = '../train.txt'  # Replace with the actual path to your text file

arabic_text_data:str
with open(file_path, 'r', encoding='utf-8') as file:
    arabic_text_data = file.read()
end_time = time.time()

# Tokenize Arabic text
print(str(end_time-start_time) + ': Tokenizing...')
tokenized_arabic_data = [word_tokenize(sentence) for sentence in arabic_text_data.split('.')]
end_time = time.time()

# Train Word2Vec model
print(str(end_time-start_time) + ': Training Word2Vec...')
arabic_model = Word2Vec(sentences=tokenized_arabic_data, vector_size=100, window=5, min_count=1, workers=4)
end_time = time.time()

# Save model
print(str(end_time-start_time) + ': Saving model')
arabic_model.save("word2vec_arabic_model.bin")

# Access word vectors
# vector = arabic_model.wv['اخْتَلَفَتْ']
# similar_words = arabic_model.wv.most_similar('اخْتَلَفَتْ', topn=20)

# Example new word
new_word = 'اختلفت'

# Tokenize the new word (you can use your own tokenizer)
tokenized_new_word = word_tokenize(new_word)
print(tokenized_new_word)

# Infer vector for the new word
vector = arabic_model.wv.get_mean_vector(tokenized_new_word)

# Find top-N similar words to the inferred vector
similar_words = arabic_model.wv.similar_by_vector(vector, topn=5)

print(vector)

out:str = ''
for word in similar_words:
    out += word[0]
    out += ' '

with open('similar.txt', 'w', encoding='utf-8') as file:
    file.write(out)