"""
Model Training Script for Toxic Comments Classification

This script trains a machine learning model to classify toxic comments
using the Toxic Comment Classification Challenge dataset.

The model uses TF-IDF vectorization and Logistic Regression for multi-label classification.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import warnings
warnings.filterwarnings('ignore')


# Toxicity categories
TOXICITY_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# File paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'toxic_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')


def preprocess_text(text):
    """
    Clean and preprocess text data with enhanced techniques to better differentiate
    between safe and toxic content.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()
    
    # Handle common greetings and innocent text patterns first
    # Preserve common friendly greetings
    text = re.sub(r'\bh+i+\b', 'hi', text)  # Normalize multiple i's in greetings like "hii", "hiii"
    text = re.sub(r'\bh+e+l+o+\b', 'hello', text)  # Normalize "heelllooo" -> "hello"
    text = re.sub(r'\by+o+\b', 'yo', text)  # Normalize "yyoo" -> "yo"
    
    # Expand contractions to capture more toxic patterns
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Handle common toxic abbreviations
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bur\b", "your", text)
    text = re.sub(r"\bcant\b", "cannot", text)
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\bdoesnt\b", "does not", text)
    text = re.sub(r"\bwont\b", "will not", text)

    # Capture repeated characters that may indicate emphasis in toxic comments
    # But be more selective - only normalize if there are 3+ repetitions
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Replace triple+ repetitions with double

    # Keep important punctuation that might be relevant for toxicity detection
    # But remove most special characters and digits
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def create_sample_dataset():
    """
    Create a sample dataset for training if real dataset is not available
    This is for demonstration purposes
    
    Returns:
        pd.DataFrame: Sample dataset with comments and labels
    """
    print("\n📝 Creating sample dataset for demonstration...")
    
    # Sample toxic and non-toxic comments
    sample_data = {
        'comment_text': [
            # Non-toxic comments
            "This is a great article, thank you for sharing!",
            "I completely agree with your point of view.",
            "Very informative and well written.",
            "Thanks for the clarification, this helps a lot.",
            "Interesting perspective, I learned something new.",
            "Great discussion, everyone makes valid points.",
            "I appreciate the detailed explanation.",
            "This is exactly what I was looking for.",
            "Well researched and presented.",
            "Thank you for contributing to this topic.",
            "I enjoyed reading this, very insightful.",
            "Good points made throughout the article.",
            "Appreciate the effort put into this research.",
            "Helpful and well-structured content.",
            "This adds valuable insights to the topic.",
            "Constructive criticism is welcome here.",
            "Looking forward to more articles like this.",
            "The methodology used here is sound.",
            "Evidence-based approach is commendable.",
            "Respectful discourse is important in this community.",
            
            # Mild toxic comments
            "I don't really agree with this.",
            "This is not very helpful.",
            "I have some concerns about this approach.",
            "Perhaps there's a better way to do this.",
            "I'm skeptical about these claims.",
            "This doesn't seem entirely accurate.",
            "There might be some issues here.",
            "I think there are better alternatives.",
            "Not convinced by this argument.",
            "This could be improved in several ways.",
            
            # Toxic comments
            "You are such an idiot, this is completely wrong!",
            "This is the stupidest thing I've ever read.",
            "What a terrible article, you should be ashamed.",
            "You're a fool if you believe this nonsense.",
            "This is absolute garbage and a waste of time.",
            "I hate people who think like this.",
            "You're completely worthless and your opinion doesn't matter.",
            "This is the dumbest argument ever made.",
            "Shut up, you don't know what you're talking about.",
            "You're an absolute moron for saying this.",
            "You are so incredibly stupid and wrong.",
            "This is pathetic and embarrassing to read.",
            "Complete trash, waste of bandwidth.",
            "You have no clue what you're doing.",
            "Absolutely ridiculous and offensive content.",
            
            # Severe toxic
            "I hope you die for posting this garbage.",
            "Someone should kill you for being this stupid.",
            "You deserve to suffer for this idiocy.",
            "Die in a fire, nobody wants your garbage.",
            "You should be eliminated from society.",
            "Hope something terrible happens to you.",
            
            # Obscene
            "This damn post is complete bullshit.",
            "What the hell is wrong with you?",
            "This shit is fucking ridiculous.",
            "Damn it, this crap is annoying.",
            "You fucking moron, learn to write.",
            "Go to hell with your stupid opinions.",
            "This is fucking stupid and worthless.",
            "You piece of shit, learn something.",
            
            # Threats
            "I'm going to find you and hurt you.",
            "Watch your back, you'll regret this.",
            "I will hunt you down and destroy you.",
            "You better watch out or else something bad will happen.",
            "You'll pay for saying these things.",
            "I know where you live, be careful.",
            "This won't go unpunished, I promise.",
            "Mark my words, you'll face consequences.",
            
            # Insults
            "You're a complete and total failure.",
            "What an incompetent fool you are.",
            "You are absolutely useless.",
            "You're nothing but a pathetic loser.",
            "You're the worst person I've ever seen.",
            "Completely incompetent and clueless.",
            "Total waste of space and oxygen.",
            "Incredibly dense and unintelligent.",
            
            # Identity hate
            "I hate your entire group of people.",
            "People like you are ruining everything.",
            "Your race is inferior and should be eliminated.",
            "I despise everyone from your community.",
            "People of your kind are a plague.",
            "Your entire demographic is problematic.",
            "You represent everything wrong with society.",
            "Your group has caused nothing but trouble."
        ],
        'toxic': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
        'severe_toxic': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
        'obscene': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
        'threat': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
        'insult': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,0,0,1,1,0,1,1,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0],
        'identity_hate': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Augment dataset by duplicating samples to make it larger
    df = pd.concat([df] * 10, ignore_index=True)
    
    print(f"✓ Created sample dataset with {len(df)} samples")
    
    return df


def train_model(df=None):
    """
    Train the toxic comment classification model with improved pipeline
    
    Args:
        df (pd.DataFrame, optional): Training dataset. If None, creates sample dataset.
    """
    print("\n" + "="*60)
    print("🤖 TOXIC COMMENTS CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load or create dataset
    if df is None:
        df = create_sample_dataset()
    
    # Preprocess text data
    print("\n🔧 Preprocessing text data...")
    df['cleaned_text'] = df['comment_text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"✓ Preprocessed {len(df)} comments")
    
    # Prepare features and labels
    X = df['cleaned_text'].values
    y = df[TOXICITY_LABELS].values
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(X)}")
    for label in TOXICITY_LABELS:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Split data into train and test sets
    print("\n✂️  Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # Create TF-IDF Vectorizer with better parameters for accuracy
    print("\n🔤 Creating TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,      # Increased features for better representation
        min_df=3,               # Minimum document frequency (higher to reduce noise)
        max_df=0.7,             # Maximum document frequency (lower to reduce common terms)
        ngram_range=(1, 3),     # Use unigrams, bigrams, and trigrams
        strip_accents='unicode',
        stop_words='english',
        sublinear_tf=True       # Apply sublinear scaling for better performance
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"✓ Vectorizer created with {min(len(vectorizer.get_feature_names_out()), 10000)} features")
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model using One-vs-Rest Logistic Regression with tuned parameters
    print("\n🏋️  Training the model...")
    print("   Algorithm: Logistic Regression with One-vs-Rest strategy")
    
    # Using a more robust approach with balanced class weights
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,                  # Regularization strength (reduced to prevent overfitting)
            solver='liblinear',     # More stable solver for smaller datasets
            max_iter=1000,          # Maximum iterations
            random_state=42,
            class_weight='balanced', # Handle class imbalance
            n_jobs=1                # Prevent threading issues on Windows
        )
    )
    
    model.fit(X_train_tfidf, y_train)
    print("✓ Model training completed!")
    
    # Make predictions on test set
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics for each label
    print("\n📊 Performance Metrics:")
    print("-" * 60)
    print(f"{'Category':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for i, label in enumerate(TOXICITY_LABELS):
        accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        
        print(f"{label:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("-" * 60)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f}")
    
    # Save the model and vectorizer
    print("\n💾 Saving model and vectorizer...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {MODEL_PATH}")
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved to: {VECTORIZER_PATH}")
    
    # Test with sample predictions
    print("\n🧪 Testing with sample predictions...")
    test_samples = [
        "This is a great article!",
        "You are an idiot and a fool!",
        "I will find you and hurt you.",
        "This damn post is complete bullshit.",
        "Your race is inferior and should be eliminated.",
        "Thanks for the helpful information.",
        "I respectfully disagree with your point."
    ]
    
    for sample in test_samples:
        cleaned = preprocess_text(sample)
        sample_vec = vectorizer.transform([cleaned])
        prediction = model.predict_proba(sample_vec)[0]
        
        print(f"\n   Text: '{sample}'")
        print(f"   Predictions:")
        for label, pred in zip(TOXICITY_LABELS, prediction):
            if pred > 0.1:  # Show predictions above 10% confidence
                print(f"      - {label}: {pred:.4f}")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n💡 You can now run the Flask API server with: python app.py")
    print()


if __name__ == "__main__":
    # Note: In production, you would load the actual Toxic Comment Classification dataset
    # from Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    # 
    # To use the real dataset:
    # 1. Download train.csv from Kaggle
    # 2. Place it in the same directory as this script
    # 3. Uncomment the following code:
    #
    # if os.path.exists('train.csv'):
    #     print("Loading dataset from train.csv...")
    #     df = pd.read_csv('train.csv')
    #     train_model(df)
    # else:
    #     print("train.csv not found. Using sample dataset...")
    #     train_model()
    
    # For now, we'll use the sample dataset
    train_model()