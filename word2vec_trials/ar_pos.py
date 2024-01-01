# ###########################   FAILED  ########################

import spacy

# Load the Arabic language model
nlp = spacy.load("xx_ent_wiki_sm")

# Example Arabic text
arabic_text = "اللغة العربية جميلة وغنية بالتنوع."

# Process the text
doc = nlp(arabic_text)

# Access part-of-speech tags
for token in doc:
    print(f"Word: {token.text}, POS Tag: {token.pos_}")
