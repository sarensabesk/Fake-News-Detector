# 📰 Fake News Detector  

A machine learning–powered system that classifies news articles as **REAL** or **FAKE** using **natural language processing (NLP)** and **Support Vector Machines (SVMs)**.  

This project includes both the **model training pipeline** and a **Flask-powered web application** with an HTML interface so users can test the detector in their browser.  

---

## 🚀 Features  
- **Automated Fake News Classification** → instantly determine if an article is REAL or FAKE.  
- **Flask Web App** → clean web interface for easy predictions.  
- **High Accuracy** → achieves **~90% accuracy** on test data.  
- **TF-IDF Vectorization** → converts text into meaningful features by weighing important words.  
- **LinearSVC Model** → efficient, scalable classifier for large text datasets.  
- **Custom Predictions** → works on any text input (via website or script).  

---

## 📊 Model Performance  
- **Accuracy on test set**: ~90%  
- Handles thousands of articles efficiently.  
- Correctly identifies fake clickbait-style headlines while maintaining strong performance on genuine news.  

**Example Predictions:**  
- *“Breaking: Aliens Land in New York”* → **FAKE**  
- *“Central Bank Announces Interest Rate Cut”* → **REAL**  

---

## 🧑‍💻 How It Works  
1. **Data Preprocessing**  
   - Loads labeled dataset (`fake_or_real_news.csv`).  
   - Converts labels → numeric (REAL = 0, FAKE = 1).  

2. **Feature Engineering**  
   - Text transformed into TF-IDF features.  
   - Common words and stopwords filtered out.  

3. **Model Training**  
   - Uses **Linear Support Vector Classifier (LinearSVC)**.  
   - Trained on 80% of the dataset, tested on 20%.  

4. **Evaluation**  
   - Accuracy score printed after training.  
   - Predictions can be run on any new article.  

5. **Flask**
   - run python app.py
   - Go to: http://127.0.0.1:5000/

---

## 🔮 Future Directions  
- Explore **deep learning models** (e.g., Transformers, LSTMs).  
- Expand dataset for broader generalization.  
- Deploy as a **browser extension** to flag articles directly online.  

---

## 🙌 Acknowledgments  
- [scikit-learn](https://scikit-learn.org/) → machine learning algorithms.  
- [Pandas](https://pandas.pydata.org/) → data handling.  
- [NumPy](https://numpy.org/) → numerical computations.  

---

✨ *This project demonstrates that even a lightweight ML pipeline can achieve strong accuracy in detecting fake news.*  
