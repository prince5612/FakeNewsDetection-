
import os
import pickle
import json
import re
import requests
import numpy as np
from datetime import datetime
import nltk
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import pymongo
import urllib.parse
from dotenv import load_dotenv
load_dotenv()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ------------------------------
# MongoDB Atlas Setup for Feedback
# ------------------------------

username = os.getenv('MONGO_USER')
password = os.getenv('MONGO_PASSWORD')
# URL-encode username and password
username_esc = urllib.parse.quote_plus(username)
password_esc = urllib.parse.quote_plus(password)
# MONGO_URI = "mongodb+srv://fakenews456781:FakeNews@5612@cluster0.xii2h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # 
MONGO_URI = f"mongodb+srv://{username_esc}:{password_esc}@cluster0.xii2h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Replace with your MongoDB Atlas URI
client = pymongo.MongoClient(MONGO_URI)
db = client["news_feedback"]
feedback_collection = db["feedbacks"]

# ------------------------------
# Global Configuration
# ------------------------------
MODEL_SAVE_PATH = "saved_model"
TOKENIZER_SAVE_PATH = "saved_tokenizer"
PREPROCESS_CACHE = "preprocessed_data.pkl"
MAX_LENGTH = 96
BATCH_SIZE = 48

# ------------------------------
# Initialize Stopwords and Lemmatizer
# ------------------------------
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# ------------------------------
# Preprocessing Cache
# ------------------------------
if os.path.exists(PREPROCESS_CACHE):
    with open(PREPROCESS_CACHE, 'rb') as f:
        preprocess_cache = pickle.load(f)
else:
    preprocess_cache = {}

def save_preprocessing_cache():
    with open(PREPROCESS_CACHE, 'wb') as f:
        pickle.dump(preprocess_cache, f)

def preprocess_text(text):
    if text in preprocess_cache:
        return preprocess_cache[text]
    text_nopunct = re.sub(r'[^\w\s]', '', text)
    text_nonum = re.sub(r'\d+', '', text_nopunct)
    text_lower = text_nonum.lower()
    words = nltk.word_tokenize(text_lower)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    processed = ' '.join(words[:50])
    preprocess_cache[text] = processed
    return processed

# ------------------------------
# Model & Tokenizer Initialization
# ------------------------------
def load_or_initialize_model():
    config_path = os.path.join(MODEL_SAVE_PATH, "config.json")
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                json.load(f)
            model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
            tokenizer = BertTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)
            print("Loaded saved model and tokenizer.")
        except Exception as e:
            print("Error loading saved model configuration:", e)
            print("Reinitializing model and tokenizer from scratch.")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    else:
        print("No saved model found. Initializing new model and tokenizer.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return model, tokenizer

model, tokenizer = load_or_initialize_model()

# ------------------------------
# Text Encoding for BERT
# ------------------------------
def encode_texts(texts, labels):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks), np.array(labels)

# ------------------------------
# Trusted News Sources
# ------------------------------
TRUSTED_NEWS_SOURCES = [
    'reuters.com',
    'apnews.com',
    'bbc.com',
    'nytimes.com',
    'theguardian.com',
    'wsj.com',
    'cnn.com',
    'aljazeera.com',
    'ndtv.com',
    'thehindu.com',
    'hindustantimes.com',
    'indianexpress.com',
    'washingtonpost.com',
    'npr.org',
    'forbes.com',
    'bloomberg.com',
    'theatlantic.com',
    'economist.com',
    'ft.com',
    'usatoday.com',
    'abcnews.go.com',
    'cbsnews.com',
    'nbcnews.com',
    'news.yahoo.com',
    'scmp.com',
    'straitstimes.com',
    'globalnews.ca',
    'cbc.ca',
    'lemonde.fr',
    'dw.com',
    'france24.com',
    'rtve.es',
    'elpais.com',
    'japantimes.co.jp',
    'smh.com.au',
    'theage.com.au',
    'theaustralian.com.au',
    'theglobeandmail.com',
    'torontosun.com',
    'thestar.com',
    'timesofindia.indiatimes.com',
    'livemint.com',
    'moneycontrol.com',
    'cnbc.com',
    'marketwatch.com',
    'economictimes.indiatimes.com',
    'business-standard.com'
]

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

def fetch_trusted_news(query):
    site_filter = " OR ".join([f"site:{site}" for site in TRUSTED_NEWS_SOURCES])
    search_url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&"
        f"q={query} {site_filter}&num=7"
    )
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        results = response.json().get('items', [])
        return [
            {
                "title": item.get('title', ''),
                "link": item.get('link', ''),
                "snippet": item.get('snippet', ''),
                "source": urlparse(item.get('link')).netloc
            }
            for item in results
        ]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def calculate_content_similarity(input_text, articles):
    vectorizer = TfidfVectorizer(stop_words='english')
    input_vector = vectorizer.fit_transform([input_text])
    similarities = []
    for article in articles:
        article_text = f"{article['title']} {article['snippet']}"
        if len(article_text) < 50:
            continue
        article_vector = vectorizer.transform([article_text])
        similarity = cosine_similarity(input_vector, article_vector)[0][0]
        if similarity >= 0.4:
            similarities.append({
                "source": article['source'],
                "similarity": round(similarity, 2),
                "link": article['link']
            })
    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)

# ------------------------------
# Prediction Function
# ------------------------------
def predict_news(text):
    cleaned_text = preprocess_text(text)
    save_preprocessing_cache()
    input_ids, attention_masks, _ = encode_texts([cleaned_text], [0])
    outputs = model.predict([input_ids, attention_masks], batch_size=1)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    ml_prediction = np.argmax(probabilities)
    predicted_label = "Real News" if ml_prediction == 0 else "Fake News"
    ml_confidence = probabilities[ml_prediction]
    details = [f"ML Confidence: {ml_confidence:.2f}, Prediction: {predicted_label}"]

    score = 50
    if ml_prediction == 0:
        score += 40 * ml_confidence
    else:
        score -= 30 * ( ml_confidence)

    articles = fetch_trusted_news(text)
    similarities = calculate_content_similarity(text, articles)
    penalty = 0
    if not similarities:
        penalty += 30
        details.append("⚠️ No trusted source matches (-30)")
    else:
        best_match = similarities[0]
        match_score = best_match['similarity']
        if match_score >= 0.75:
            score += 40
            details.append(f"✅ Strong trusted match: {best_match['source']} (+40)")
        elif match_score >= 0.60:
            score += 20
            details.append(f"⚠️ Moderate trusted match: {best_match['source']} (+20)")
        else:
            penalty += 20
            details.append(f"❌ Weak trusted match: {best_match['source']} (-20)")
    final_score = max(0, min(100, score - penalty))
    if final_score >= 75:
        verdict = "High Confidence: Real News ✅"
    elif final_score >= 55:
        verdict = "Likely Real News ⚠️"
    elif final_score >= 45:
        verdict = "Unverified Claim ⚠️"
    elif final_score >= 30:
        verdict = "Suspected Fake News ❌"
    else:
        verdict = "High Confidence: Fake News ❌"
    return {
        "verdict": verdict,
        "score": int(final_score),
        "details": details,
        "matches": similarities
    }

# ------------------------------
# Feedback Storage Function
# ------------------------------
def store_feedback(text, predicted_verdict, correct_label, feedback_details, score):
    feedback_doc = {
        "text": text,
        "predicted_verdict": predicted_verdict,
        "correct_label": correct_label,
        "feedback_details": feedback_details,
        "score": score,
        "timestamp": datetime.utcnow()
    }
    feedback_collection.insert_one(feedback_doc)
    print("Feedback stored in MongoDB.")

# ------------------------------
# Retraining Function from Feedback
# ------------------------------
def retrain_model_with_feedback():
    feedbacks = list(feedback_collection.find())
    if not feedbacks:
        print("No feedback data found for retraining.")
        return
    texts = []
    labels = []
    for fb in feedbacks:
        texts.append(fb["text"])
        # Convert "Real News" to label 0 and "Fake News" to label 1
        labels.append(0 if fb["correct_label"].lower().startswith("real") else 1)
    preprocessed_texts = [preprocess_text(text) for text in texts]
    input_ids, attention_masks, labels_arr = encode_texts(preprocessed_texts, labels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    )
    print("Retraining model on feedback data...")
    model.fit(
        [input_ids, attention_masks],
        labels_arr,
        epochs=1,
        batch_size=BATCH_SIZE
    )
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    print("Model retrained on feedback and saved.")
    
    # After successful retraining, delete the feedback entries from the database
    delete_result = feedback_collection.delete_many({})
    print(f"Deleted {delete_result.deleted_count} feedback entries from MongoDB.")
