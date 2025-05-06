## 1. Introduction to NLP

- **Definition**: Science of developing systems that understand and generate human language
- **Significance**: Human language is a unique, complex communication tool that distinguishes human intelligence
- **Challenge**: Creating effective language representations for machines despite human children's remarkable efficiency in language learning
- **Modern Approach**: Deep learning tools to solve the representation problem

## 2. Word Representation Methods

### 2.1 Signifiers and Signified

- Words are signs (signifiers) representing concepts (signified)
- Word meaning depends on context (e.g., "tea" can refer to a specific cup or general concept)
- Words have semantic relationships (e.g., "coffee" and "tea" are beverages)
- Central NLP challenge: Representing complex meanings using discrete symbols

### 2.2 One-Hot Vector Representation

- **Approach**: Represent each word as a unique vector with a single '1'
- **Example**: In vocabulary of size V, each word is a V-dimensional vector
- **Limitation**: Fails to encode similarity between words (all vectors are equidistant)

### 2.3 Annotated Discrete Properties

- **Approach**: Represent words using manually annotated linguistic features
- **Sources**: WordNet, UniMorph for annotated linguistic information
- **Example**: "tea" → [plural noun, hyponym-of-beverage, synonym-of-chai]
- **Limitations**:
    - Limited vocabulary coverage
    - High dimensionality issues
    - Outperformed by data-driven approaches

## 3. Distributional Semantics and Word2Vec

### 3.1 Core Principle

- Firth's hypothesis: "You shall know a word by the company it keeps"
- Word meaning reflected in surrounding words (context)
- Similar words have similar distributions of surrounding words

### 3.2 Co-occurrence Matrices

- **Construction**: Count how often words appear together in context
- **Context Types**:
    - Small windows (1-5 words): Capture syntactic relationships
    - Document-level: Capture semantic/topic relationships
- **Limitations**:
    - High dimensionality (size of vocabulary)
    - Overemphasis on frequent words
- **Improvements**: Log frequency transformation
- **Implementation Example**:

```python
import numpy as np
from collections import defaultdict

def create_cooccurrence_matrix(documents):
    # Step 1: Determine vocabulary
    vocabulary = set()
    for document in documents:
        vocabulary.update(document.split())
    
    # Convert vocabulary to a dictionary mapping words to indices
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    
    # Step 2: Create a matrix of zeros
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    
    # Step 3: Count co-occurrences in each document
    for document in documents:
        words = document.split()
        # For each word in the document
        for word in words:
            word_idx = word_to_idx[word]
            # Add counts for all other words in the document
            for other_word in words:
                if other_word != word:  # Don't count word with itself
                    other_idx = word_to_idx[other_word]
                    cooccurrence_matrix[word_idx, other_idx] += 1
    
    # Step 4: Normalize rows
    row_sums = cooccurrence_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = cooccurrence_matrix / row_sums
    
    return normalized_matrix, word_to_idx, idx_to_word

# Example usage
documents = [
    "It's hot and delicious. I poured the tea for my uncle.",
    "The tea was too hot to drink immediately.",
    "My uncle enjoys hot tea with sugar."
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
```

### 3.3 Word2Vec Model

- **Approach**: Learn low-dimensional vectors by predicting context words
- **Architecture**: Skip-gram predicts outside context words (O) given center word (C)
- **Components**:
    - Each word has two vectors: u (as center) and v (as context)
    - Softmax function calculates p(o|c) based on vector dot products
- **Training Objective**: Minimize negative log-likelihood of true context words
- **Advantages**: Captures semantic relationships in dense vector space

### 3.4 Word2Vec Training Process

- **Loss Function**: Sum of negative log probabilities across word-context pairs
- **Optimization**: Stochastic Gradient Descent (SGD)
    - Update vectors in direction opposite to gradient
    - Use learning rate (α) to control step size
    - Initialize vectors randomly with small variance
- **Efficiency**: Use mini-batches for faster training

## Key Comparison of Representation Methods

|Method|Advantages|Disadvantages|
|---|---|---|
|One-Hot Vectors|Simple implementation|No semantic information; high dimensionality|
|Annotated Properties|Incorporates linguistic knowledge|Limited coverage; human effort intensive|
|Co-occurrence Matrices|Captures word relationships|High dimensionality; frequency bias|
|Word2Vec|Dense, meaningful representations; learns from data|Computationally intensive training|
