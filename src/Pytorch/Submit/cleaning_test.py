from Globals import harakat, classes
from utils import *

def clean_text(text):
  '''
  Clean Date Set
  Remove all except Arabic, Punctutation,Diacretics
  '''
  # Removing HTML tags
  cleaned_text = remove_html_tags(text)

  # Remove URLs
  # cleaned_text = re.sub(r'http\S+', '', cleaned_text)
  # another way
  cleaned_text = remove_urls(cleaned_text)

  # Remove special Arabic character (Kashida)
  cleaned_text = cleaned_text.replace('\u0640', '')

  # Separate Numbers
  # cleaned_text = re.sub(r'(\d+)', r' \1 ', cleaned_text)

  # Remove Multiple Whitespaces
  cleaned_text = remove_white_spaces(cleaned_text)
  # cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()


  # Clear Punctuations
  cleaned_text = clear_punctuations(cleaned_text)

  # Remove english letters and english and arabic numbers
  cleaned_text = clear_english_and_numbers(cleaned_text)
  # content = remove_english_letters(content)

  # Remove shifts
  cleaned_text = remove_shift_j(cleaned_text)

  # surrond them with spaces
  # content = fix_numbers(content)

  return cleaned_text

def remove_html_tags(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()

def remove_urls(content):
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                     ' ', content, flags=re.MULTILINE)
    content = re.sub(r'www(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                     ' ', content, flags=re.MULTILINE)
    return content

def remove_white_spaces(content):
    content = re.sub(r'[^\S\n]*\n[\s]*', '\n', content, flags=re.MULTILINE)
    content = re.sub(r'[^\S\n]+', ' ', content, flags=re.MULTILINE)
    content = re.sub(r'\A | \Z', '', content, flags=re.MULTILINE)
    return content

def clear_punctuations(text):
    # All Punctutation Marks
    all_punctuation = set(string.punctuation)
    print("all_punctuation",all_punctuation)

    # Exclude 
    exclude_chars = {'.', ',','،', '؛',':',"?","؟",'«','»'}

    # Include Extra Charaters
    # all_punctuation.add()

    # Create a new set without the excluded characters
    filtered_punctuation = all_punctuation - exclude_chars

    # Convert the set back to a string
    filtered_punctuation_string = ''.join(filtered_punctuation)

    # print(filtered_punctuation_string)
    text = "".join(c for c in text if c not in filtered_punctuation_string)
    return text

def clear_english_and_numbers(text):
    text = re.sub(r"[a-zA-Z0-9٠-٩]", " ", text)
    return text

def remove_shift_j(content):
    return content.replace('ـ', '')



def get_sentences(data,split_regex="[\n,،]+"):
    # Replace multiple spaces with a single space using regular expression
    # cleaned_sentence = re.sub(' +', ' ', sentence)
    
    return [re.sub(' +', ' ', sent) for line in re.split(split_regex, data) if line for sent in sent_tokenize(line.strip()) if sent]
    #return [sent for line in data.split('\n') if line for sent in sent_tokenize(line) if sent]



def clear_tashkel(text,array_format=False):
  
    harakat_list=[    
      harakat["Fatha"],
      harakat["Fathatan"],
      harakat["Damma"],
      harakat["Dammatan"],
      harakat["Kasra"], 
      harakat["Kasratan"],
      harakat["Sukun"],
      harakat["Shadda"]
      ]
    DIACRITICS_LIST = [chr(ord(code)) for code in harakat_list]
    clr_text=text.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))
    if(array_format):
      return [char for char in clr_text]
    return clr_text
    # text = "".join(c for c in text if ord(c) not in harakat)
    # return text


import sys
import unicodedata
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('--model', help='Specify the model1', required=True)
    
    args = parser.parse_args()

    model= args.model

    print("Model",model)


    input_file_path=f"/content/drive/MyDrive/Kaggle/test/test_no_diacritics.txt"
    # Read data from the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    print(data[:10])


    # Step 1: Cleaning Process
    print("Cleaning .....")
    cleaned_text = clean_text(data)

    # print(cleaned_text)

    # output_file_path=f"/content/drive/MyDrive/Kaggle/{model}/clean_step(1).txt"
    # # Open the output file for writing with UTF-8 encoding
    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     # Write the data to the output file
    #     output_file.write(cleaned_text)
    print("Done Cleaning :D")


    # Step 2: Tokenization (using nltk sent_tokenize)
    split_regex="[\n.,،؛:«»?؟]+"
    print("Tokenization to Sentences",split_regex,".....")
    sentences = get_sentences(data=cleaned_text,split_regex=split_regex)

    # output_file_path=f"/content/drive/MyDrive/Kaggle/{model}/clean_step(2).txt"
    # # Open the output file for writing with UTF-8 encoding
    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     # Write each element from the list on a new line
    #     for sentence in sentences:
    #         output_file.write(str(sentence) + '\n')
    print("Done Tokenizing :D")


    # Step 4: Tashkel Removal
    print("Removing Tashkel ......")
    # remove may be faster
    sentences_without_tashkel = [clear_tashkel(sentence,array_format=True) for sentence in sentences]
    
    output_file_path=f"/content/drive/MyDrive/Kaggle/{model}/X.pickle"
    # Save the NumPy array to a pickle file
    with open(output_file_path, 'wb') as file:
      pickle.dump(sentences_without_tashkel, file)

    # Open the file in write mode
    with open(f"/content/drive/MyDrive/Kaggle/{model}/X.txt", 'w') as file:
        # Write each element of the list to a new line
        for item in sentences_without_tashkel:
            file.write(str(item) + '\n')

    # sentences_without_tashkel = [clear_tashkel(sentence,array_format=False) for sentence in sentences_with_fixed_diacritization]
    # output_file_path="/content/drive/MyDrive/Tashkeel/train_clean_step(4).txt"
    # Open the output file for writing with UTF-8 encoding
    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     # Write each element from the list on a new line
    #     for sentence in sentences_without_tashkel:
    #         output_file.write(str(sentence) + '\n')
    print("Preparing X Done :D",len(sentences_without_tashkel),"Sentences")
   

    print(sentences_without_tashkel[0])

