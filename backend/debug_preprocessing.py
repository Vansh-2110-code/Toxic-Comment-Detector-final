import sys
sys.path.append('.')
from train_model import preprocess_text
import pickle
import os

# Test preprocessing
test_cases = ['hii', 'hello', 'hi', 'hiii', 'heelllo', 'hiiii', 'yoo']
print("Preprocessing results:")
for text in test_cases:
    processed = preprocess_text(text)
    print(f"'{text}' -> '{processed}'")

# Load the vectorizer to check what features are being extracted
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')

if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Transform the preprocessed texts
    processed_texts = [preprocess_text(text) for text in test_cases]
    vectors = vectorizer.transform(processed_texts)
    
    print("\nVector representations:")
    for i, text in enumerate(test_cases):
        print(f"'{text}' -> vector shape: {vectors[i].shape}")
        # Show some feature values
        dense_vec = vectors[i].toarray()[0]
        non_zero_features = sum(dense_vec > 0)
        print(f"  Non-zero features: {non_zero_features}/{len(dense_vec)}")
else:
    print("Vectorizer not found!")