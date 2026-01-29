"""Vercel API Route for Toxic Comment Classification"""

import json
import pickle
import os
import re
import numpy as np
from datetime import datetime


# Define global variables (these will be loaded once when the function is first called)
model = None
vectorizer = None
TOXICITY_LABELS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

def load_models():
    """Load the pre-trained model and vectorizer from disk"""
    global model, vectorizer
    
    # In Vercel environment, files are located relative to the function root
    # The models should be available in the deployed package
    model_paths = [
        './backend/models/toxic_classifier.pkl',
        '../backend/models/toxic_classifier.pkl',
        '../../backend/models/toxic_classifier.pkl',
        './models/toxic_classifier.pkl',
        '../models/toxic_classifier.pkl',
        'models/toxic_classifier.pkl'
    ]
    
    vectorizer_paths = [
        './backend/models/vectorizer.pkl',
        '../backend/models/vectorizer.pkl',
        '../../backend/models/vectorizer.pkl',
        './models/vectorizer.pkl',
        '../models/vectorizer.pkl',
        'models/vectorizer.pkl'
    ]

    try:
        model_path = None
        vectorizer_path = None
        
        # Find the correct model path
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        # Find the correct vectorizer path
        for path in vectorizer_paths:
            if os.path.exists(path):
                vectorizer_path = path
                break

        if model_path and vectorizer_path:
            print(f"Loading model from: {model_path}")
            print(f"Loading vectorizer from: {vectorizer_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Models loaded successfully!")
        else:
            print("Model files not found, using demo mode")
            print(f"Tried model paths: {model_paths}")
            print(f"Tried vectorizer paths: {vectorizer_paths}")
            model = None
            vectorizer = None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        model = None
        vectorizer = None


def preprocess_text(text):
    """
    Preprocess input text for model prediction
    """
    # Convert to lowercase
    text = text.lower()
    
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


def get_demo_prediction(text):
    """
    Generate demo predictions when model is not available
    Uses simple keyword matching for demonstration purposes
    """
    text_lower = text.lower()
    
    # Simple keyword-based detection for demo
    toxic_keywords = ['hate', 'stupid', 'idiot', 'kill', 'die', 'worst']
    severe_keywords = ['kill', 'die', 'murder']
    obscene_keywords = ['damn', 'hell']
    threat_keywords = ['kill', 'hurt', 'attack', 'destroy']
    insult_keywords = ['stupid', 'idiot', 'fool', 'dumb']
    hate_keywords = ['hate']
    
    # Calculate simple scores based on keyword presence
    predictions = {
        'toxic': 0.8 if any(word in text_lower for word in toxic_keywords) else 0.1,
        'severe_toxic': 0.7 if any(word in text_lower for word in severe_keywords) else 0.05,
        'obscene': 0.6 if any(word in text_lower for word in obscene_keywords) else 0.08,
        'threat': 0.75 if any(word in text_lower for word in threat_keywords) else 0.04,
        'insult': 0.65 if any(word in text_lower for word in insult_keywords) else 0.07,
        'identity_hate': 0.6 if any(word in text_lower for word in hate_keywords) else 0.03
    }
    
    return predictions


def predict_toxicity(text):
    """
    Predict toxicity levels for input text
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # If model is available, use it for prediction
    if model is not None and vectorizer is not None:
        try:
            # Vectorize the text
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Check if we have any features - if not, it's likely a simple greeting
            dense_vec = text_vectorized.toarray()[0]
            non_zero_features = sum(dense_vec > 0)
            
            # If no features found, treat as very low toxicity (likely a greeting)
            if non_zero_features == 0:
                # Return low toxicity scores for simple greetings
                return {
                    'toxic': 0.1,
                    'severe_toxic': 0.05,
                    'obscene': 0.05,
                    'threat': 0.05,
                    'insult': 0.1,
                    'identity_hate': 0.05
                }
            
            # Get predictions (probabilities for better results)
            predictions = model.predict_proba(text_vectorized)[0]
            
            # Create result dictionary
            result = {
                label: float(pred) 
                for label, pred in zip(TOXICITY_LABELS, predictions)
            }
            
            return result
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return get_demo_prediction(text)
    else:
        # Use demo prediction
        return get_demo_prediction(text)


def get_toxicity_level(score):
    """
    Categorize toxicity score into levels
    """
    if score < 0.2:
        return 'Safe'
    elif score < 0.4:
        return 'Moderate'
    elif score < 0.6:
        return 'Toxic'
    else:
        return 'Highly Toxic'


def handler(event, context):
    """Vercel/Serverless function handler"""
    global model, vectorizer
    
    # Log incoming request for debugging
    print(f"Received request: {event.get('httpMethod')} {event.get('path')}")
    
    # Load models if not already loaded
    if model is None or vectorizer is None:
        print("Loading models...")
        load_models()
    
    # Get the HTTP method
    http_method = event.get('httpMethod', '').upper()
    
    # Handle OPTIONS request for CORS
    if http_method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': ''
        }
    
    # Handle GET request (health check)
    if http_method == 'GET':
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'online',
                'service': 'Toxic Comments Classification API',
                'version': '1.0.0',
                'model_loaded': model is not None,
                'message': 'API is running' if model is not None else 'API is running in demo mode',
                'timestamp': datetime.now().isoformat()
            })
        }
    
    # Handle POST request (prediction)
    if http_method == 'POST':
        try:
            # Parse the request body
            request_body = event.get('body')
            if request_body is None:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': False,
                        'error': 'Request body is missing.'
                    })
                }
            
            # Handle both string and already-parsed JSON bodies
            if isinstance(request_body, str):
                try:
                    body = json.loads(request_body)
                except json.JSONDecodeError:
                    return {
                        'statusCode': 400,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps({
                            'success': False,
                            'error': 'Invalid JSON in request body'
                        })
                    }
            else:
                body = request_body
            
            # Validate input
            if 'text' not in body:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': False,
                        'error': 'No text provided. Please send JSON with "text" field.'
                    })
                }
            
            text = body['text']
            
            # Validate text is not empty
            if not text or len(text.strip()) == 0:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': False,
                        'error': 'Text cannot be empty.'
                    })
                }
            
            # Get predictions
            predictions = predict_toxicity(text)
            
            # Determine if comment is toxic (slightly higher threshold to reduce false positives)
            is_toxic = any(score > 0.65 for score in predictions.values())
            max_toxicity = max(predictions.values())
            
            # Prepare response
            response = {
                'success': True,
                'text': text,
                'predictions': predictions,
                'is_toxic': is_toxic,
                'max_toxicity': round(max_toxicity, 4),
                'toxicity_level': get_toxicity_level(max_toxicity),
                'timestamp': datetime.now().isoformat(),
                'demo_mode': model is None
            }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response)
            }
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': f'An error occurred: {str(e)}'
                })
            }
    
    # Method not allowed
    return {
        'statusCode': 405,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'success': False,
            'error': 'Method not allowed'
        })
    }