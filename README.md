# sms-spam-detection
# 📩 SMS Spam Detection using Machine Learning

A complete end‑to‑end **SMS Spam Classifier** built using Python and Machine Learning. This project detects whether a given SMS message is **Spam** or **Ham (Not Spam)** using NLP techniques and multiple ML algorithms. It includes data cleaning, EDA, preprocessing, vectorization, model comparison, ensemble learning, and deployment‑ready artifacts.

---

## 🚀 Project Highlights

* ✅ Data Cleaning & Duplicate Removal
* ✅ Exploratory Data Analysis (EDA)
* ✅ Text Preprocessing (Tokenization, Stopwords, Stemming)
* ✅ Feature Engineering (Characters, Words, Sentences)
* ✅ TF‑IDF Vectorization
* ✅ Model Training with Multiple Algorithms
* ✅ Performance Comparison (Accuracy & Precision)
* ✅ Ensemble Models (Voting & Stacking)
* ✅ Best Model Selection
* ✅ Model Serialization using Pickle
* ✅ Deployment Ready

---

## 🧠 Algorithms Used

The following machine learning models were trained and evaluated:

* Naive Bayes (Gaussian, Multinomial, Bernoulli)
* Logistic Regression
* Support Vector Classifier (SVC)
* K‑Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* AdaBoost
* Bagging Classifier
* Extra Trees Classifier
* Gradient Boosting
* XGBoost
* Voting Classifier (Ensemble)
* Stacking Classifier (Ensemble)

The best performing model was:

> ⭐ **TF‑IDF + Multinomial Naive Bayes**
> Accuracy ≈ **97%**
> Precision ≈ **100%**

---

## 📊 Dataset

* Dataset: `spam.csv`
* Source: UCI SMS Spam Collection
* Total Samples: **5,169 (after cleaning)**
* Classes:

  * Ham (0)
  * Spam (1)

---

## ⚙️ Workflow

1. Data Loading
2. Cleaning & Removing Duplicates
3. Exploratory Data Analysis
4. Text Preprocessing
5. Feature Extraction using TF‑IDF
6. Train/Test Split
7. Model Training
8. Evaluation
9. Ensemble Learning
10. Saving Model & Vectorizer

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* NLTK
* Scikit‑learn
* Matplotlib
* Seaborn
* WordCloud
* Pickle
* Google Colab / Jupyter Notebook

---

## 🧪 Example Prediction

```python
def predict_spam(msg):
    transformed = transform_text(msg)
    vector = tfidf.transform([transformed])
    result = mnb.predict(vector)[0]
    return "Spam" if result == 1 else "Not Spam"

predict_spam("Congratulations! You won a lottery")
```

---

## 📁 Project Structure

```
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── notebook.ipynb
├── README.md
```

---

## 🌐 Future Improvements

* Build a Web App using Flask / Streamlit
* Deploy on Render / HuggingFace
* Add REST API
* Improve preprocessing using Lemmatization
* Add deep learning models (LSTM / BERT)

---

## 👨‍💻 Author

**Shahid Bashir**
Assistant Professor | Machine Learning & Networks
YouTube: *Rahil Tech Hacks*

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork, improve, and contribute.

---

> This project is built for learning, research, and real‑world spam detection s
