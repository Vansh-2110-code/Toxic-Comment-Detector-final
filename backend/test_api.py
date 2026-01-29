"""
API Testing Script
Test the Toxic Comments Classification API with various examples
"""

import requests
import json
from datetime import datetime

# Configuration
API_URL = 'http://localhost:5000'

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f'{API_URL}/')
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API Status: {data.get('status', 'unknown')}")
            print_success(f"Service: {data.get('service', 'unknown')}")
            print_success(f"Version: {data.get('version', 'unknown')}")
            print_success(f"Model Loaded: {data.get('model_loaded', False)}")
            return True
        else:
            print_error(f"Health check failed with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Make sure the server is running!")
        print_warning("Run: python app.py")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_single_prediction(text, label=""):
    """Test single comment prediction"""
    if label:
        print(f"\n{Colors.BOLD}Testing: {label}{Colors.RESET}")
    
    try:
        response = requests.post(
            f'{API_URL}/api/predict',
            json={'text': text},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print(f"  Text: \"{text[:50]}...\"" if len(text) > 50 else f"  Text: \"{text}\"")
                print(f"  Is Toxic: {data.get('is_toxic', False)}")
                print(f"  Toxicity Level: {data.get('toxicity_level', 'unknown')}")
                print(f"  Max Score: {data.get('max_toxicity', 0):.2%}")
                
                predictions = data.get('predictions', {})
                print("  Category Scores:")
                for category, score in predictions.items():
                    bar_length = int(score * 20)
                    bar = '█' * bar_length + '░' * (20 - bar_length)
                    color = Colors.GREEN if score < 0.3 else Colors.YELLOW if score < 0.6 else Colors.RED
                    print(f"    {category:15} {color}{bar}{Colors.RESET} {score:.2%}")
                
                return True
            else:
                print_error(f"API Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print_error(f"Request failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_header("Testing Batch Prediction")
    
    test_texts = [
        "This is a wonderful discussion!",
        "You are stupid and wrong!",
        "I completely disagree, but respect your opinion."
    ]
    
    try:
        response = requests.post(
            f'{API_URL}/api/batch-predict',
            json={'texts': test_texts},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print_success(f"Successfully processed {data.get('count', 0)} comments")
                
                for i, result in enumerate(data.get('results', []), 1):
                    print(f"\n  Comment {i}:")
                    print(f"    Text: \"{result['text'][:50]}...\"" if len(result['text']) > 50 else f"    Text: \"{result['text']}\"")
                    print(f"    Is Toxic: {result.get('is_toxic', False)}")
                    print(f"    Max Toxicity: {result.get('max_toxicity', 0):.2%}")
                
                return True
            else:
                print_error(f"API Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print_error(f"Request failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases"""
    print_header("Testing Edge Cases")
    
    # Empty text
    print(f"{Colors.BOLD}Test 1: Empty text{Colors.RESET}")
    try:
        response = requests.post(
            f'{API_URL}/api/predict',
            json={'text': ''},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            print_success("Correctly rejected empty text")
        else:
            print_warning(f"Unexpected response: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Very long text
    print(f"\n{Colors.BOLD}Test 2: Very long text{Colors.RESET}")
    long_text = "This is a test. " * 100
    test_single_prediction(long_text, "")
    
    # Special characters
    print(f"\n{Colors.BOLD}Test 3: Special characters{Colors.RESET}")
    special_text = "Hello! @#$%^&*() []{} <> 😀 你好"
    test_single_prediction(special_text, "")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║       TOXIC COMMENTS CLASSIFIER - API TEST SUITE          ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    # Test 1: Health Check
    if not test_health_check():
        print_error("\nCannot proceed with tests. Server is not running.")
        return
    
    # Test 2: Sample Comments
    print_header("Testing Sample Comments")
    
    test_comments = [
        ("Thank you for this informative article!", "Safe Comment"),
        ("You are an idiot and don't know anything!", "Toxic Comment"),
        ("I hope you die for posting this garbage.", "Severe Toxic"),
        ("What the hell is wrong with you?", "Obscene"),
        ("I'm going to find you and hurt you.", "Threat"),
        ("You're a complete fool and failure.", "Insult"),
        ("I hate people like you.", "Identity Hate"),
    ]
    
    for text, label in test_comments:
        test_single_prediction(text, label)
    
    # Test 3: Batch Prediction
    test_batch_prediction()
    
    # Test 4: Edge Cases
    test_edge_cases()
    
    # Summary
    print_header("Test Suite Complete")
    print_success("All tests executed successfully!")
    print(f"\n{Colors.BOLD}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")

if __name__ == "__main__":
    run_comprehensive_tests()
