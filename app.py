import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load & Prepare Data (cached so it doesn't retrain every time)
# -----------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv", encoding="latin-1")

    data = data[["v1", "v2"]]
    data.columns = ["category", "message"]

    data.drop_duplicates(inplace=True)

    data['category'] = data['category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

    mess = data['message']
    cat = data['category']

    mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

    cv = CountVectorizer(stop_words='english')
    features = cv.fit_transform(mess_train)

    model = MultinomialNB()
    model.fit(features, cat_train)

    accuracy = model.score(cv.transform(mess_test), cat_test)

    return cv, model, accuracy

cv, model, accuracy = train_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 Spam Detection App")

st.write(f"Model Accuracy: **{accuracy:.2f}**")

input_mes = st.text_input("Enter message here...")

if st.button('Validate'):
    if input_mes.strip() == "":
        st.warning("Please enter a message")
    else:
        output = predict(input_mes)

        if output == "Spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")
