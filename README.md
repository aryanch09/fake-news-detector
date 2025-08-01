# 📰 Fake News Detector

A machine learning-powered web app that classifies news headlines as **real** or **fake** using a logistic regression model trained on Kaggle's "Fake and Real News" dataset.

Built with:
- Python
- Scikit-learn
- Streamlit
- Joblib

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/aryan-c09/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

Note: To run the app, first run train_and_save_model.py to generate the model files (lr_model.pkl, vectorizer.pkl).
You can get the dataset from (https://drive.google.com/drive/folders/1ByadNwMrPyds53cA6SDCHLelTAvIdoF_) or use your own.
