import numpy as np
from collections import defaultdict


def create_cooccurrence_matrix(documents):
    
    vocabulary = set()
    for doc in documents:
        words = doc.split() # Split the document into words
        vocabulary.update(words) # Add words to the vocabulary set
    
    # Create a mapping from words to indices
    # Create a mapping from indices to words
    word_to_idx = {word: i for i, word in enumerate(vocabulary)} 
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}

    vocab_size = len(vocabulary)
    

    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    print(f"Co-occurrence Matrix Shape: {cooccurrence_matrix.shape}")
    
    for doc in documents:
        words = doc.split()
        
        for word in words:
            word_idx = word_to_idx[word]
            
            for other_word in words:
                if word != other_word:
                    other_idx = word_to_idx[other_word]
                    cooccurrence_matrix[word_idx, other_idx] += 1
                    print(cooccurrence_matrix)

    # Normalize the rows
    row_sums = cooccurrence_matrix.sum(axis=1, keepdims=True)
    normalized_cooccurrence_matrix = cooccurrence_matrix / row_sums
   
    return normalized_cooccurrence_matrix, word_to_idx, idx_to_word
    




documents = [
    "It's hot and delicious. I poured the tea for my uncle.",
    "The tea was too hot to drink immediately.",
    "My uncle enjoys hot tea with sugar.",
    "I prefer my tea with milk.",
    "The tea leaves are fresh and fragrant.",
    "Tea is a popular beverage around the world.",
    "I like to drink tea in the morning.",
    "Tea can be enjoyed hot or cold.",
    "My uncle brews the tea with care.",
    "I often drink tea while reading a book.",
    "Tea is known for its health benefits.",
    "I like to add lemon to my tea.",
    "Tea is a great way to relax.",
]
cooccurrence_matrix, word_to_idx, idx_to_word = create_cooccurrence_matrix(documents)


# Print embedding for 'tea'
if 'tea' in word_to_idx:
    tea_idx = word_to_idx['tea']
    tea_embedding = cooccurrence_matrix[tea_idx]
    print(f"Embedding for 'tea':")
    for i, value in enumerate(tea_embedding):
        if value > 0:
            print(f"  {idx_to_word[i]}: {value:.4f}")


# Summary of the co-occurrence matrix
# It is a square matrix where each row and column corresponds to a word in the vocabulary.
# The value at position (i, j) in the matrix represents the co-occurrence count of word i with word j.
# The matrix is symmetric, meaning that the co-occurrence of word i with word j is the same as the co-occurrence of word j with word i.
# The matrix is normalized by dividing each element by the sum of the elements in its row,
# which gives a probability distribution for the co-occurrence of words.
# The resulting matrix can be used to find similar words based on their co-occurrence patterns.
# The co-occurrence matrix can be used in various NLP tasks, such as word embeddings, topic modeling, and semantic analysis.