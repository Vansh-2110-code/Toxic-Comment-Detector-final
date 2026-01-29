import requests
import json

# Test more examples to check accuracy
examples = [
    'This is very helpful, thank you!',
    'You suck and this is terrible!',
    'I disagree with your opinion.',
    'Go die in a hole you worthless human.',
    'Great job on this article!',
    'You are an absolute moron.',
    'I appreciate the insight.',
    'Kill yourself you scum.'
]

print("Testing model accuracy improvements:\n")

for text in examples:
    response = requests.post('http://localhost:5000/api/predict', json={'text': text})
    result = response.json()
    print(f'Text: {text}')
    print(f'Toxic: {result["is_toxic"]}, Max Toxicity: {result["max_toxicity"]:.4f}')
    print(f'Toxicity Level: {result["toxicity_level"]}')
    print('-' * 50)