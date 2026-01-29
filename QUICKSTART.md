# Toxic Comments Classifier - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend Server
```bash
cd backend
python app.py
```
The server will start at `http://localhost:5000`

### Step 3: Open the Frontend
Simply open `frontend/index.html` in your web browser!

---

## 📝 Optional: Train the Model

The app works in demo mode without training. To train with sample data:

```bash
cd backend
python train_model.py
```

This will create a trained model in the `backend/models/` directory.

---

## 🧪 Testing the Application

### Web Interface
1. Open `frontend/index.html`
2. Click one of the sample buttons
3. Click "Analyze Comment"
4. View the results!

### API Testing
```bash
# Test with curl
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This is a great article!\"}"
```

---

## 🛠️ Troubleshooting

### Backend won't start?
- Make sure Python 3.8+ is installed: `python --version`
- Check if all dependencies are installed: `pip list`
- Try installing dependencies again: `pip install -r requirements.txt --upgrade`

### Frontend can't connect to backend?
- Make sure backend is running on port 5000
- Check browser console (F12) for errors
- Verify the API_BASE_URL in `frontend/script.js` matches your backend URL

### CORS errors?
- The backend is configured to allow all origins for development
- If issues persist, check your browser's CORS settings

---

## 📊 What to Expect

### Demo Mode (default)
- Uses simple keyword-based detection
- Works immediately without training
- Good for testing the UI/UX

### Trained Model Mode
- After running `train_model.py`
- Uses machine learning (Logistic Regression)
- More accurate predictions
- Based on TF-IDF features

---

## 🎯 Next Steps

1. **Use Real Dataset**: Download from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
2. **Improve Model**: Experiment with different algorithms
3. **Deploy**: Host on cloud platforms (Heroku, AWS, etc.)
4. **Integrate**: Use the API in your own applications

---

## 📚 Documentation

See `README.md` for complete documentation including:
- Detailed API documentation
- Model training guide
- Architecture overview
- Configuration options

---

**Happy Moderating! 🛡️**
