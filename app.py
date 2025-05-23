import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the saved model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords + stemming
    return ' '.join(words)

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter a message to check whether it's Spam or Not Spam.")

# User input
user_input = st.text_area("Enter the message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess and predict
        cleaned_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)

        # Output result
        if prediction[0] == 1:
            st.error("❌ This is a Spam message.")
        else:
            st.success("✅ This is a Ham (Not Spam) message.")
