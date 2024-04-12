import numpy as np
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

country_regions = {
    'Africa': ['Senegal', 'Egypt', 'South Africa', 'Cameroon', 'Nigeria', 'Ethiopia', 'Kenya', 'Morocco', 'Ghana', 'Angola', 'Tunisia', 'Libya', 'Sudan', 'Uganda', 'Ivory Coast', 'Mali', 'Somalia', 'Zimbabwe', 'Tanzania', 'Zambia', 'Mozambique', 'Rwanda', 'Madagascar', 'Sierra Leone', 'Liberia', 'Guinea', 'Burkina Faso', 'Niger', 'Chad', 'Congo', 'Mauritania', 'Namibia', 'Botswana', 'Swaziland', 'Lesotho', 'Equatorial Guinea', 'Gabon', 'Eritrea', 'Djibouti', 'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Mayotte', 'Reunion', 'Western Sahara'],
    'US': ['United States'],
    'China': ['China', 'Taiwan', 'Hong Kong', 'Macau'],
    'EU': ['France', 'Italy', 'Germany', 'United Kingdom', 'Spain', 'Netherlands', 'Greece', 'Portugal', 'Belgium', 'Sweden', 'Austria', 'Denmark', 'Finland', 'Ireland', 'Czech Republic', 'Romania', 'Poland', 'Hungary', 'Slovak Republic', 'Luxembourg', 'Bulgaria', 'Croatia', 'Slovenia', 'Lithuania', 'Latvia', 'Estonia', 'Malta', 'Cyprus', 'Monaco'],
    'Russia': ['Russia'],
    'Ukraine': ['Ukraine'],
    'Middle East': ['Egypt', 'Iran', 'Saudi Arabia', 'Iraq', 'United Arab Emirates', 'Syria', 'Yemen', 'Israel', 'Jordan', 'Lebanon', 'Palestine', 'Oman', 'Kuwait', 'Qatar', 'Bahrain', 'Turkey', 'Cyprus']
}


def map_country_to_region(category):
  """
  Maps a given country to its corresponding region based on a predefined dicitionary.

  Parameters:
    category (str): The name of the country.
  Returns:
    str: The region to which the conutry belongs, or None if not found.
  """

  for region, countries in country_regions.items():
    if category in countries:
      return region
  return None

def create_bar_plot(data, title, xlabel, ylabel):

    """
    Plot the top and bottom entries of a DataFrame based on the count of occurrences of a column

    Parameters:
        data (pandas.Series or DataFrame): Data to visualize.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """

    plt.figure(figsize=(10,6))
    data.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_sentiment_distribution_for_top_sources(rating_df, top_10_sources, sentiment_column="title_sentiment"):
    """
    Plots sentiment distribution (positive, neutral, negative) for the top 10 sources in a multi-plot grid.

    Args:
        rating_df (pandas.DataFrame): The DataFrame containing sentiment data.
        top_10_sources (list): A list of the top 10 source names to plot.
        sentiment_column (str, optional): The name of the column containing sentiment data. Defaults to "title_sentiment".
    """

    # Filter data for top 10 sources
    filtered_df = rating_df[rating_df['source_name'].isin(top_10_sources)]

    # Calculate sentiment distribution
    sentiment_by_source = filtered_df.groupby('source_name')[sentiment_column].value_counts().unstack(fill_value=0)

    # Define colors for each sentiment
    sentiment_colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}

    # Create a single figure with multiple subplots
    nrows, ncols = 2, 5  # Adjust if needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    # Iterate through top 10 sources and plot on subplots
    source_index = 0
    for row in range(nrows):
        for col in range(ncols):
            if source_index >= len(sentiment_by_source):
                break
            source_name = sentiment_by_source.index[source_index]
            sentiment_counts = sentiment_by_source.iloc[source_index]
            sentiment_counts.plot(kind='bar', stacked=False, ax=axes[row, col], title=source_name,
                                  color=sentiment_counts.index.map(sentiment_colors))
            axes[row, col].set_xlabel("Sentiment")
            axes[row, col].set_ylabel("Count")  # Assuming counts, adjust if proportion
            source_index += 1

    # Adjust layout
    fig.suptitle(f"Sentiment Distribution for Top 10 Source Names", fontsize=16)
    plt.tight_layout()
    plt.show()

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocesses text for further analysis.

    Args:
        text: The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """

    # Remove URLs
    url_pattern = r"(https?://)?(www\.)?(\S+\.\S+)(/\S*)?"
    text = re.sub(url_pattern, "", text)

    # Remove punctuations
    punctuation_pattern = r"[^\w\s]"
    text = re.sub(punctuation_pattern, "", text)

    # Remove stop words
    words = text.split()
    filtered_word = [word for word in words if word.lower() not in stop]
    text = " ".join(filtered_word)

    normalized_text = lemma.lemmatize(text)



    return normalized_text


def fit_tokenizer(train_sentences, num_words, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences

    Args:
        train_sentences (list of string): lower-cased sentences without stopwords to be used for training
        num_words (int) - number of words to keep when tokenizing
        oov_token (string) - symbol for the out-of-vocabulary token

    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """

    # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
    tokenizer = Tokenizer(num_words = num_words , oov_token = oov_token)

    # Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer
     

def seq_and_pad(sentences, tokenizer, padding, maxlen):
    """
    Generates an array of token sequences and pads them to the same length

    Args:
        sentences (list of string): list of sentences to tokenize and pad
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        padding (string): type of padding to use
        maxlen (int): maximum length of the token sequence

    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences using the correct padding and maxlen
    padded_sequences = pad_sequences(sequences , maxlen = maxlen , padding = padding)

    return padded_sequences


def tokenize_labels(all_labels, split_labels):
    """
    Tokenizes the labels

    Args:
        all_labels (list of string): labels to generate the word-index from
        split_labels (list of string): labels to tokenize

    Returns:
        label_seq_np (array of int): tokenized labels
    """

    # Instantiate the Tokenizer (no additional arguments needed)
    label_tokenizer = Tokenizer()

    # Fit the tokenizer on all the labels
    label_tokenizer.fit_on_texts(all_labels)

    # Convert labels to sequences
    label_seq = label_tokenizer.texts_to_sequences(split_labels)

    # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
    label_seq_np = np.array(label_seq)
    label_seq_np = np.subtract(label_seq_np , 1)


    return label_seq_np