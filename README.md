# 🚀 Automated Blog Tag Generator (NLP)

An AI-powered web application that automatically generates relevant tags for blog content using Natural Language Processing (NLP) and multi-label text classification.

This project improves content organization, discoverability, and SEO by intelligently analyzing blog text and suggesting meaningful tags.

---

## ✨ Features

* 🧠 NLP-based automatic tag prediction
* 🏷️ Multi-label classification (one blog → multiple tags)
* ⚖️ Class-balanced Logistic Regression model
* 🔤 TF-IDF with n-grams for better context understanding
* 📁 TXT file upload support with robust encoding handling
* 🌙 Dark mode toggle
* 📊 Tag confidence visualization
* ⚡ Streamlit premium interactive UI
* 🧩 Modular and clean project structure

---

## 🧠 Problem Statement

Manually tagging blog posts is time-consuming and often inconsistent. This project automates the tagging process using machine learning to:

* Improve content discoverability
* Enhance SEO performance
* Maintain tagging consistency
* Reduce manual effort

---

## 🏗️ Project Architecture

```
Blog Text
   ↓
Text Preprocessing (NLTK)
   ↓
TF-IDF Vectorization
   ↓
Multi-Label Classifier (OneVsRest + Logistic Regression)
   ↓
Predicted Tags + Confidence
   ↓
Streamlit Web Interface
```

---

## 🛠️ Tech Stack

**Languages & Libraries**

* Python
* Scikit-learn
* NLTK
* Pandas
* NumPy
* Streamlit

**ML Techniques**

* TF-IDF Vectorization
* N-grams (1,2)
* MultiLabelBinarizer
* One-vs-Rest Classification
* Class balancing

---

## 📁 Project Structure

```
blog-auto-tagging/
│
├── data/
│   └── blogs.csv
│
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── mlb.pkl
│
├── utils.py
├── train.py
├── streamlit_app.py
└── README.md
```

---

## 🧪 Example Usage

**Input**

```
Artificial intelligence is transforming healthcare and medical diagnosis.
```

**Output**

```
AI, Healthcare
```

---

## 📊 Model Details

| Component      | Choice                     |
| -------------- | -------------------------- |
| Vectorizer     | TF-IDF                     |
| N-grams        | (1,2)                      |
| Classifier     | Logistic Regression        |
| Strategy       | One-vs-Rest                |
| Problem Type   | Multi-label classification |
| Class Handling | Balanced weights           |

---

## 🚀 Future Improvements

* 🔥 BERT/Transformer-based tagging
* 📄 PDF and DOCX support
* 🌐 Cloud deployment
* 📊 Advanced analytics dashboard
* 🧠 Hybrid keyword + ML tagging
* 🏷️ Auto tag suggestion ranking
* 🌍 Multi-language support

---

## 🎯 Learning Outcomes

This project demonstrates practical skills in:

* Natural Language Processing (NLP)
* Text preprocessing and tokenization
* Multi-label text classification
* Feature engineering with TF-IDF
* Model balancing and prediction tuning
* Streamlit web application development
* End-to-end ML pipeline design

---

## 👩‍💻 Author

**Gauri Murathe**

If you found this project useful, consider giving it a ⭐ on GitHub!

---

## 📜 License

This project is intended for educational and research purposes.
