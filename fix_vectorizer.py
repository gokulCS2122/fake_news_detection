import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Sample text data to fit the vectorizer
sample_data = [
    "Breaking news on politics and economy",
    "This is a fake news article",
    "The stock market is going up",
    "COVID-19 vaccine approved by government",
    "Local sports team wins championship"
]

# Step 2: Fit the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(sample_data)

# Step 3: Save the fitted vectorizer to 'vectorizer.pkl'
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Fixed and saved fitted vectorizer.pkl")
