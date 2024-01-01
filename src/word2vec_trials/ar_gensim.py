from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Example Arabic text data
arabic_text_data = "نصك العربي يذهب هنا. يمكن أن يكون سلسلة واحدة أو قائمة من السلاسل."

# Tokenize Arabic text
tokenized_arabic_data = [word_tokenize(sentence) for sentence in arabic_text_data.split('.')]

# Train Word2Vec model
arabic_model = Word2Vec(sentences=tokenized_arabic_data, vector_size=100, window=5, min_count=1, workers=4)

# Access word vectors
vector = arabic_model.wv['واحدة']
similar_words = arabic_model.wv.most_similar('واحدة', topn=5)

print(vector)
print(similar_words)