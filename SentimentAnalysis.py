import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re
from bs4 import BeautifulSoup
import sys

plt.style.use('ggplot')

# Read in data
print("Reading data...")
df = pd.read_csv(r'E:\SentimentAnalysis\SentimentAnalysis\Reviews.csv')  # Update the path to your CSV file
print("Data read successfully!")
df = df.head(5000)  # Use only the first 500 rows for this tutorial
print("Data head:\n", df.head())

# Preprocessing step to clean the text data
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['Text'] = df['Text'].apply(clean_text)
print("Text cleaned!")

# Quick EDA
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# Basic NLTK Operations
example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# VADER Sentiment Analysis
print("Initializing VADER...")
sia = SentimentIntensityAnalyzer()

# Run VADER sentiment analysis on the dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

print("VADER analysis complete!")
vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Plotting results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# RoBERTa Pretrained Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Compare Results
sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
plt.show()

# Function to analyze individual sentences
def analyze_sentence(sentence):
    cleaned_sentence = clean_text(sentence)
    vader_result = sia.polarity_scores(cleaned_sentence)
    roberta_result = polarity_scores_roberta(cleaned_sentence)
    print("VADER Analysis:", vader_result)
    print("RoBERTa Analysis:", roberta_result)

# Example usage:

while True:
    sentence = input("Enter a sentence for sentiment analysis (Press Enter to exit): ")
    if sentence.strip() == "":
        print("Exiting...")
        break
    analyze_sentence(sentence)
