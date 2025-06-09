from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Define paths to model and vectorizer
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Load the vectorizer
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Vectorizer loaded successfully.")
except Exception as e:
    vectorizer = None
    print(f"❌ Error loading vectorizer: {e}")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template("prediction.html")

    if not model or not vectorizer:
        return render_template("prediction.html", prediction_text="⚠️ Error: Model or Vectorizer not loaded.")

    try:
        news_input = request.form['news']
        print(f"📝 User input: {news_input}")

        # Transform the input and make prediction
        transformed_input = vectorizer.transform([news_input])
        prediction = model.predict(transformed_input)[0]
        print(f"📢 Prediction result: {prediction}")

        label = "🟢 The news is REAL." if prediction == 1 else "🔴 The news is FAKE."
        return render_template("prediction.html", prediction_text=label)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template("prediction.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
