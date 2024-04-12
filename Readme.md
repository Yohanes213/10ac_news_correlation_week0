# News Correlation Analysis
This repository contains the code and documentation for the News Correlation Analysis project. The project aims to perform exploratory data analysis (EDA), statistical analysis, sentiment analysis, topic modeling, and more on a dataset comprising news articles from various sources.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Task Overview](#task-overview)
- [Task 1: Project Setup and EDA](#task-1-project-setup-and-eda)
- [Task 2: Data Science Component Building](#task-2-data-science-component-building)
- [Task 3: PostgreSQL](#task-3-postgresql)
- [Task 4: Dashboards](#task-4-dashboards)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project focuses on analyzing news data to uncover insights such as top news websites, traffic analysis, sentiment analysis, topic modeling, and more. The dataset includes information about news articles, including source, author, content, sentiment, and publication date. By leveraging various data science techniques, we aim to gain actionable insights into the news landscape.

## Project Structure

The project is structured as follows:

- .github
  - workflows
    - flake8_check.yml
    - unittests.yml
    - docstring_tests.yml
- .vscode
  - settings.json
- model
  - saved_model_weights.h5
- notebooks
  - news_correlation.ipynb
- src
  - csv_handler.py
  - database.py
  - loader.py
  - main.py
  - utils.py
- tests
  - __init__.py
- .gitignore
- README.md
- app.py
- confif.json
- requirements.txt
  

## Task Overview

The project is divided into multiple tasks:

1. **Project Setup and EDA**: Setup Python environment, perform exploratory data analysis, and answer specific questions about the data.
2. **Data Science Component Building**: Develop components for machine learning operations (MLOps), conduct time series analysis, classification of headlines, topic modeling, sentiment analysis, and predictive modeling.
3. **PostgreSQL**: Design database schema, load data into PostgreSQL, and utilize it for storing ML features.
4. **Dashboards**: Design and implement a dashboard using Streamlit or React to visualize analysis results.
5. **Deployment**: Deploy the project using GitHub Actions for continuous deployment, and configure environment variables and PostgreSQL database.

## Task 1: Project Setup and EDA

- **Git and GitHub Setup**: Created a GitHub repository and set up version control.
- **Python Environment Setup**: Prepared a Python environment for data analysis.
- **Exploratory Data Analysis (EDA)**: Analyzed the dataset to answer various questions about news articles, including top websites, traffic analysis, sentiment analysis, and more.

## Task 2: Data Science Component Building

- **Topic Modeling**: Implemented topic modeling on news articles to uncover underlying themes.
- **Sentiment Analysis**: Conducted sentiment analysis on news article titles to understand public perception.
- *Add more details if needed*

## Task 3: PostgreSQL

- **Database Schema Design**: Designed a schema for PostgreSQL to store ML features.
- **Data Loading**: Loaded data from CSV into PostgreSQL database for efficient storage and retrieval.
- *Add more details if needed*

## Task 4: Dashboards

- **Streamlit Dashboard**: Designed and implemented a Streamlit dashboard to visualize EDA and model prediction results.
- *Add more details if needed*

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for any suggestions, bug fixes, or enhancements.

## License

This project is licensed under the [MIT License](LICENSE).
