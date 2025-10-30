# src/data_preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources (only first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Remove URLs, special characters, numbers, and extra spaces."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)            # keep only alphabets
    text = re.sub(r"\s+", " ", text).strip()         # remove extra spaces
    return text

def preprocess(df, text_col="Review"):
    """Perform deduplication, cleaning, tokenization, lemmatization, stopword removal."""
    df = df.drop_duplicates(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def tokenize_lemmatize(text):
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
        return " ".join(tokens)

    df["clean_text"] = df[text_col].apply(tokenize_lemmatize)

    # Handle missing or noisy data
    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].fillna("neutral")
    else:
        df["Sentiment"] = "neutral"

    df = df[df["clean_text"].str.strip() != ""]
    return df

def main():
    # Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_path = os.path.join(data_dir, "Dataset-SA.csv")
    output_path = os.path.join(data_dir, "cleaned_feedback.csv")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(input_path)

    # Rename likely text column if needed
    if "feedback" in df.columns:
        text_col = "feedback"
    elif "review" in [c.lower() for c in df.columns]:
        text_col = [c for c in df.columns if c.lower() == "review"][0]
    else:
        text_col = df.columns[0]

    print(f"Using text column: {text_col}")
    cleaned_df = preprocess(df, text_col)

    # Save cleaned dataset
    os.makedirs(data_dir, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")
    print(f"Total records after cleaning: {len(cleaned_df)}")

if __name__ == "__main__":
    main()
