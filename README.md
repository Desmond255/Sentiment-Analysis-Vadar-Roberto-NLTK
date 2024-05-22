# Sentiment-Analysis-Vadar-Roberto-NLTK
It is an NLTK based sentiment analysis model which uses VADAR and ROBERTO models in Python and judges the sentiment.

# Sentiment Analysis Project

This project performs sentiment analysis on Amazon reviews using VADER and RoBERTa models. It also provides visualizations of the sentiment scores and allows for interactive sentiment analysis of individual sentences.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [License](#license)

## Introduction
This project leverages natural language processing (NLP) techniques to analyze the sentiment of Amazon reviews. It uses:
- **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media.
- **RoBERTa** (A Robustly Optimized BERT Pretraining Approach), a transformer model that has been pre-trained on a large corpus of Twitter data for sentiment analysis.

##Code Overview
Import Libraries
The script starts by importing the necessary libraries for data manipulation, visualization, NLP, and using pre-trained models.

Read Data
Reads the CSV file containing Amazon reviews and selects the first 500 rows for analysis.

Preprocessing Function
Defines a function to clean the text data by removing HTML tags, URLs, special characters, and converting text to lowercase.

Exploratory Data Analysis (EDA)
Visualizes the distribution of review scores using bar plots.

Basic NLTK Operations
Performs tokenization, part-of-speech tagging, and named entity recognition on a sample text.

VADER Sentiment Analysis
Initializes the VADER sentiment analyzer and applies it to the dataset.

Plotting VADER Results
Visualizes the VADER sentiment scores across different review ratings using bar plots.

RoBERTa Model
Loads the pre-trained RoBERTa model for sentiment analysis and defines a function to get sentiment scores.

Apply RoBERTa Analysis
Applies both VADER and RoBERTa sentiment analysis to the dataset and stores the results.

Compare Results
Creates pair plots to compare VADER and RoBERTa sentiment scores.

Interactive Sentiment Analysis
Defines a function for interactive sentiment analysis of individual sentences using both models and provides a loop for continuous user input.

Usage
Run the script to see the visualizations and sentiment analysis results on the sample dataset.
Enter individual sentences for real-time sentiment analysis using both VADER and RoBERTa models.

## Features
- Cleans and preprocesses text data.
- Performs sentiment analysis using VADER and RoBERTa models.
- Visualizes sentiment scores across different review ratings.
- Allows for interactive sentiment analysis of individual sentences.

## Setup
Follow these steps to set up and run the project on Visual Studio 2022.

### Prerequisites
- Python 3.7 or higher
- Visual Studio 2022

### Installing Required Libraries
Open your terminal or command prompt and run the following commands:
```sh
pip install pandas numpy matplotlib seaborn nltk tqdm transformers scipy beautifulsoup4




