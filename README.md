📰 Fake News Detector

A machine learning–powered fake news detection system that classifies news articles as REAL or FAKE using natural language processing (NLP) and Support Vector Machines (SVMs).

This project demonstrates how to combine TF-IDF text vectorization with a Linear Support Vector Classifier to build a baseline model for detecting misinformation.

🚀 Features

Automated Fake News Classification: Predict whether an article is REAL or FAKE.

TF-IDF Vectorization: Converts text into meaningful numerical features, filtering out common words.

LinearSVC Model: Efficient and high-performing linear classifier for text data.

Custom Predictions: Input your own .txt news articles for instant classification.

Extensible Pipeline: Easy to improve by swapping vectorizers or models (e.g., Naive Bayes, Logistic Regression, or even Deep Learning).

📂 Dataset

The model is trained on a labeled dataset (fake_or_real_news.csv) containing thousands of news articles tagged as REAL or FAKE.

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt

🧑‍💻 Usage
Train the model & evaluate accuracy
python3 News.py


Expected output:

Accuracy: 0.92

Test on a new article

Save your article in a text file (new_article.txt) and run:

with open("new_article.txt", "r", encoding="utf-8") as f:
    new_text = f.read()

new_text_vectorized = vectorizer.transform([new_text])
prediction = clf.predict(new_text_vectorized)[0]

print("Prediction:", "REAL" if prediction == 0 else "FAKE")

📊 Example Results

Accuracy: ~90% on held-out test data.

Correctly flags fake headlines like “Breaking: Aliens Land in New York”.

Recognizes genuine reporting from major outlets.

🔮 Future Improvements

Add deep learning models (e.g., LSTMs, Transformers).

Use larger, more diverse datasets.

Build a web app interface for real-time predictions.

Deploy via Flask / FastAPI or as a browser extension.

🙌 Acknowledgments

scikit-learn
 for ML algorithms.

Pandas
 for data handling.

NumPy
 for numerical computations.

✨ This project is an educational demo that shows the potential of machine learning in fighting misinformation.