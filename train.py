import os
os.makedirs("models", exist_ok=True)

import pandas as pd
import pickle
from utils import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

print("📥 Loading dataset...")

# Load dataset
data = pd.read_csv("data/blogs.csv")
print("Dataset shape:", data.shape)

# Clean text
data["clean_content"] = data["content"].apply(preprocess)

# Convert tags to list
data["tag_list"] = data["tags"].apply(lambda x: x.split(";"))

print("🔤 Vectorizing text...")

# TF-IDF vectorization  ← THIS CREATES X
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=1
)

X = vectorizer.fit_transform(data["clean_content"])  # ⭐ IMPORTANT

# Multi-label encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data["tag_list"])

print("Classes:", mlb.classes_)

print("🤖 Training model...")

# Balanced model (better predictions)
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced")
)

model.fit(X, y)

print("💾 Saving models...")

# Save files
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(mlb, open("models/mlb.pkl", "wb"))

print("✅ Model trained and saved successfully!")