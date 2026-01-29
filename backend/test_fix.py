import requests

# Test the problematic case
response = requests.post('http://localhost:5000/api/predict', json={'text': 'hii'})
result = response.json()
print('Text: hii')
print('Toxic:', result['is_toxic'])
print('Max Toxicity:', result['max_toxicity'])
print('Toxicity Level:', result['toxicity_level'])
print('Predictions:', result['predictions'])
print()

# Also test other variations
test_cases = ['hii', 'hello', 'hi', 'hiii', 'heelllo', 'hiiii', 'yoo']
for text in test_cases:
    response = requests.post('http://localhost:5000/api/predict', json={'text': text})
    result = response.json()
    print(f'Text: {text} -> Toxic: {result["is_toxic"]}, Max Toxicity: {result["max_toxicity"]:.4f}, Level: {result["toxicity_level"]}')

print("\nTesting some clearly toxic content to ensure sensitivity is maintained:")
toxic_cases = ['you are stupid', 'this is terrible', 'go away moron']
for text in toxic_cases:
    response = requests.post('http://localhost:5000/api/predict', json={'text': text})
    result = response.json()
    print(f'Text: {text} -> Toxic: {result["is_toxic"]}, Max Toxicity: {result["max_toxicity"]:.4f}, Level: {result["toxicity_level"]}')