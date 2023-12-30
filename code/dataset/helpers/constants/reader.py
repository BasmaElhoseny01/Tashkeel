# To see the conent of these files

import pickle

# Open the file in binary mode
with open('./ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    # Load the object from the file
    data = pickle.load(file)


# Now, 'data' contains the deserialized object
print(data)
print(len(data))


# Open the file in binary mode
with open('./DIACRITICS_LIST.pickle', 'rb') as file:
    # Load the object from the file
    data = pickle.load(file)


# Now, 'data' contains the deserialized object
print(data)
print(len(data))



# Open the file in binary mode
with open('./CLASSES_LIST.pickle', 'rb') as file:
    # Load the object from the file
    data = pickle.load(file)


# Now, 'data' contains the deserialized object
print(data)
print(len(data))
