import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
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