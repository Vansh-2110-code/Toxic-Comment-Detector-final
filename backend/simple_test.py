import requests
import json

# Test the API
try:
    # Test 1: Safe comment
    response = requests.post('http://localhost:5000/api/predict', 
                           json={'text': 'This is a great article!'})
    print("Safe comment test:")
    print(json.dumps(response.json(), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Toxic comment
    response = requests.post('http://localhost:5000/api/predict', 
                           json={'text': 'You are an idiot and a fool!'})
    print("Toxic comment test:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Error: {e}")