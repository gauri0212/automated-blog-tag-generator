import pickle
from utils import preprocess

# Load saved items
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
mlb = pickle.load(open("models/mlb.pkl", "rb"))

def predict_tags(text):
    text = preprocess(text)
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tags = mlb.inverse_transform(pred)
    return tags[0]

# Test example
if __name__ == "__main__":
    blog = input("Enter blog content: ")
    print("🏷️ Predicted tags:", predict_tags(blog))