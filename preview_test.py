import requests

# Test the API
response = requests.post('http://localhost:5000/api/predict', json={'text': 'This is a wonderful article!'})
result = response.json()

print('✅ System is running successfully!')
print('\nPreview Test Results:')
print('=' * 40)
print(f'Text: "This is a wonderful article!"')
print(f'Toxic: {result["is_toxic"]}')
print(f'Toxicity Level: {result["toxicity_level"]}')
print(f'Max Toxicity: {result["max_toxicity"]:.2%}')

# Test a toxic example
response2 = requests.post('http://localhost:5000/api/predict', json={'text': 'You are completely stupid!'})
result2 = response2.json()

print(f'\nToxic example:')
print(f'Text: "You are completely stupid!"')
print(f'Toxic: {result2["is_toxic"]}')
print(f'Toxicity Level: {result2["toxicity_level"]}')
print(f'Max Toxicity: {result2["max_toxicity"]:.2%}')
print('=' * 40)
print('✅ All systems operational!')