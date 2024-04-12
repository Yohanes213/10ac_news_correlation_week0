import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Import your utility functions here
from src.utils import (
    map_country_to_region,
    preprocess_text,
    fit_tokenizer,
    seq_and_pad,
)

country_regions = {
    'Africa': ['Senegal', 'Egypt', 'South Africa', 'Cameroon', 'Nigeria', 'Ethiopia', 'Kenya', 'Morocco', 'Ghana', 'Angola', 'Tunisia', 'Libya', 'Sudan', 'Uganda', 'Ivory Coast', 'Mali', 'Somalia', 'Zimbabwe', 'Tanzania', 'Zambia', 'Mozambique', 'Rwanda', 'Madagascar', 'Sierra Leone', 'Liberia', 'Guinea', 'Burkina Faso', 'Niger', 'Chad', 'Congo', 'Mauritania', 'Namibia', 'Botswana', 'Swaziland', 'Lesotho', 'Equatorial Guinea', 'Gabon', 'Eritrea', 'Djibouti', 'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Mayotte', 'Reunion', 'Western Sahara'],

    'US': ['United States'],

    'China': ['China', 'Taiwan', 'Hong Kong', 'Macau'],

    'EU': ['France', 'Italy', 'Germany', 'United Kingdom', 'Spain', 'Netherlands', 'Greece', 'Portugal', 'Belgium', 'Sweden', 'Austria', 'Denmark', 'Finland', 'Ireland', 'Czech Republic', 'Romania', 'Poland', 'Hungary', 'Slovak Republic', 'Luxembourg', 'Bulgaria', 'Croatia', 'Slovenia', 'Lithuania', 'Latvia', 'Estonia', 'Malta', 'Cyprus', 'Monaco'],

    'Russia': ['Russia'],

    'Ukraine': ['Ukraine'],

    'Middle East': ['Egypt', 'Iran', 'Saudi Arabia', 'Iraq', 'United Arab Emirates', 'Syria', 'Yemen', 'Israel', 'Jordan', 'Lebanon', 'Palestine', 'Oman', 'Kuwait', 'Qatar', 'Bahrain', 'Turkey', 'Cyprus']
}

# Header
st.header("News Analysis Dashboard")

#st.image("https://th.bing.com/th/id/OIP.VlNBLVjksYx0JR1LU29VJQHaFj?rs=1&pid=ImgDetMain", use_column_width = True)

# Sidebar for selecting functionality (EDA or Model Prediction)
selected_option = st.sidebar.selectbox(
    "Select Functionality", ("Exploratory Data Analysis (EDA)", "Model Prediction")
)

# Data Upload Section
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['region'] = data['category'].apply(map_country_to_region)

    selected_column = st.sidebar.selectbox("Choose Column to Analyze", ['source_name', 'category', 'title_sentiment', 'region'])
else:
    st.sidebar.write("Please upload a CSV file for analysis.")
    st.stop()  # Halt execution if no file is uploaded

def perform_eda(data, selected_column):
    """Performs Exploratory Data Analysis on the loaded data"""
    if selected_column == "title_sentiment":
        analysis_data = data[selected_column].value_counts().reset_index()
        analysis_data.columns = [selected_column, "Count"]
        st.subheader(f"Sentiment Distribution of News Articles")
        fig, ax = plt.subplots()
        ax.bar(analysis_data[selected_column], analysis_data["Count"])
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
    else:
        top_data = data[selected_column].value_counts().head(10)
        bottom_data = data[selected_column].value_counts().tail(10).reset_index()
        bottom_data.columns = [selected_column, "Count"]

        st.subheader(f"Top 10 {selected_column}")
        fig, ax = plt.subplots()
        ax.bar(top_data.index, top_data.values)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        ax.set_title(f"Top 10 {selected_column}")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader(f"Bottom 10 {selected_column}")
        fig, ax = plt.subplots()
        ax.bar(bottom_data[selected_column], bottom_data["Count"])
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        ax.set_title(f"Bottom 10 {selected_column}")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

def perform_model_prediction(data):
    """Performs Model Prediction (Placeholder)"""
    data['cleaned_article'] = data['article'].apply(preprocess_text)

    tokenizer = fit_tokenizer(data['cleaned_article'], 100000, '')
    word_index = tokenizer.word_index

    text_padded_seq = seq_and_pad(data['cleaned_article'], tokenizer, 'post', 3000)

    model = load_model('model/saved_model_weights.h5')
    predictions = model.predict(text_padded_seq)

    st.write(predictions)

if st.sidebar.button("Analyze Data"):
    if selected_option == "Exploratory Data Analysis (EDA)":
        perform_eda(data, selected_column)
    else:
        perform_model_prediction(data)


# Run the main application
if __name__ == "__main__":
    pass  