import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# -------------------------------
# TRAINING SECTION
# -------------------------------

try:
    # Step 1: Load CSV file
    df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Fake-News-Detection-App-main\Fake-News-Detection-App-main\news.csv", encoding='utf-8')
    print("📋 Available columns in CSV:", df.columns.tolist())
    
    # Step 2: Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("❌ CSV file must contain 'text' and 'label' columns.")
    
    # Step 3: Extract features and labels
    X_train = df['text'].astype(str)
    y_train = df['label'].astype(str)
    
    # Step 4: Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_transformed = vectorizer.fit_transform(X_train)
    
    # Step 5: Train model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_transformed, y_train)
    
    # Step 6: Save vectorizer and model
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open("finalized_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Trained and saved model + vectorizer successfully.")

except Exception as e:
    print("❌ Error during training:", e)

# -------------------------------
# USER INPUT PREDICTION SECTION
# -------------------------------

try:
    # Step 7: Load model + vectorizer for prediction
    with open("vectorizer.pkl", "rb") as f:
        loaded_vectorizer = pickle.load(f)
    
    with open("finalized_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Step 8: Get user input and predict
    while True:
        user_news = input("\n📰 Enter news content (or type 'exit' to quit):\n")
        if user_news.strip().lower() == "exit":
            print("👋 Exiting...")
            break

        if not user_news.strip():
            print("⚠️ Please enter some news text.")
            continue

        user_input_vector = loaded_vectorizer.transform([user_news])
        result = loaded_model.predict(user_input_vector)

        print("🔎 Result:", "🟢 REAL NEWS" if result[0].upper() == "REAL" else "🔴 FAKE NEWS")

except Exception as e:
    print("❌ Error during prediction:", e)
