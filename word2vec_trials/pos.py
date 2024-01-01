import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(text):
    # Tokenize the input text
    words = word_tokenize(text)
    
    # Perform Part-of-Speech tagging
    pos_tags = pos_tag(words)
    
    return pos_tags

# Example usage
text_example = "This is a sample sentence for Part-of-Speech tagging."
result = pos_tagging(text_example)
print(result)
