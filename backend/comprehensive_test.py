import requests
import json

# Test with a wide variety of new/unseen comments to demonstrate generalization
test_cases = [
    # Completely new safe comments
    "Thank you for taking the time to explain this concept.",
    "I found this article informative and well-researched.",
    "Can you clarify this point for me?",
    "I see your perspective, though I have a different view.",
    "This helped me understand the topic better.",
    
    # New toxic comments
    "You are completely worthless and should delete this garbage.",
    "Anyone who agrees with this is a complete idiot.",
    "This is the worst trash I've ever encountered.",
    "Shut up and stop wasting everyone's time.",
    "You have no idea what you're talking about, moron.",
    
    # Ambiguous/mild disagreement
    "I respectfully disagree with your conclusion.",
    "That's an interesting point, but I think differently.",
    "I'm not convinced by this argument.",
    "There might be a better approach than this.",
    "I have some concerns about this methodology.",
    
    # Mixed sentiment
    "The beginning was good, but the ending was disappointing.",
    "Some parts were helpful, others not so much.",
    "I appreciate the effort but the quality is lacking."
]

print("Testing model generalization on new/unseen comments:\n")

for text in test_cases:
    response = requests.post('http://localhost:5000/api/predict', json={'text': text})
    result = response.json()
    print(f'Text: {text}')
    print(f'Toxic: {result["is_toxic"]}, Max Toxicity: {result["max_toxicity"]:.4f}')
    print(f'Toxicity Level: {result["toxicity_level"]}')
    
    # Show category breakdown for interesting cases
    toxic_categories = []
    for category, score in result["predictions"].items():
        if score > 0.3:  # Only show categories with significant scores
            toxic_categories.append(f"{category}: {score:.3f}")
    
    if toxic_categories:
        print(f'Significant categories: {", ".join(toxic_categories)}')
    
    print('-' * 70)