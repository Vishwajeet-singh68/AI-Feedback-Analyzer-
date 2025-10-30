# 🧠 Intelligent Customer Feedback Analysis System using AI

An **AI-powered system** designed to analyze, summarize, and predict customer sentiment from feedback data collected through various sources such as emails, chat logs, and social media comments.

This project follows a structured, modular approach — implementing all five assignment parts:
1. Data Handling  
2. Sentiment Classification Model  
3. Text Summarization  
4. Predictive Insight Generation  
5. Web App Deployment  

---

## 🚀 Features

- ✅ Upload customer feedback dataset (CSV)
- ✅ Perform **Sentiment Analysis** (Positive / Neutral / Negative)
- ✅ Generate **Short and Detailed Summaries** of long feedback
- ✅ Identify recurring issues or complaints
- ✅ Predict **Customer Satisfaction Trends**
- ✅ Visualize insights interactively
- ✅ Simple **Streamlit Web Interface**

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML/NLP Models** | DistilBERT, BART (Hugging Face Transformers) |
| **Data Processing** | Pandas, NLTK |
| **Visualization** | Plotly, Matplotlib |
| **Forecasting** | Prophet (Facebook) |

---

## 📂 Project Structure

AI-Feedback-Analyzer/
│
├── app.py # Main Streamlit app
│
├── data/
│ ├── Dataset-SA.csv # Raw dataset (input)
│ ├── cleaned_feedback.csv # Cleaned dataset (output of preprocessing)
│
├── models/
│ ├── sentiment_model.pkl # Trained Sentiment Classification Model
│ ├── summarizer.pkl # Trained Summarizer Model
│ └── insights_model.pkl # Trained Insights Model (forecasting)
│
├── src/
│ ├── data_preprocessing.py # Data cleaning and preprocessing code
│ ├── sentiment_model.py # Sentiment analysis model (DistilBERT)
│ ├── summarizer.py # Text summarizer (BART/T5)
│ ├── insights.py # Predictive insights and forecasting
│
├── requirements.txt # List of dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-Feedback-Analyzer.git
cd AI-Feedback-Analyzer
2️⃣ Create and Activate Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate       # For Windows
# or
source venv/bin/activate    # For macOS/Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🧹 Part 1 – Data Handling
Steps:
Collect or simulate 1000+ customer feedback records (Dataset-SA.csv).

Perform:

Duplicate removal

Stopword removal

Lemmatization

Tokenization

Special character cleaning

Output cleaned dataset as cleaned_feedback.csv.

Run:
bash
Copy code
python src/data_preprocessing.py
🤖 Part 2 – Sentiment Classification Model
Model Used:
DistilBERT (Transformer-based classifier)

Steps:
Load cleaned dataset

Train sentiment classifier (Positive, Negative, Neutral)

Evaluate: Accuracy, Precision, Recall, F1 Score

Save trained model as sentiment_model.pkl

Run:
bash
Copy code
python src/sentiment_model.py
📝 Part 3 – Text Summarization
Model Used:
BART (facebook/bart-large-cnn)

Steps:
Implement summarizer class (TextSummarizer)

Generate both short and detailed summaries

Save summarizer model as summarizer.pkl

Run:
bash
Copy code
python src/summarizer.py
📈 Part 4 – Predictive Insight Generation
Objectives:
Identify recurring issues or complaints

Predict customer satisfaction trends

Models:
Prophet or ARIMA

Deliverable:
Visualizations and predictions

Saved model as insights_model.pkl

Run:
bash
Copy code
python src/insights.py
🌐 Part 5 – Deployment (Streamlit App)
Features:
Upload feedback data (CSV)

Perform sentiment prediction

Generate summaries

View forecast insights

Run Application:
bash
Copy code
streamlit run app.py
Then open the provided local URL (usually http://localhost:8501).

📊 Example CSV Format
feedback
The product quality was amazing, I’ll buy again!
Service was poor and delivery was late.
The experience was okay, nothing special.

The app automatically detects columns named:

feedback

review

comments

🧾 requirements.txt
txt
Copy code
streamlit
torch
transformers
pandas
numpy
plotly
matplotlib
scikit-learn
prophet
nltk
🧰 Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named 'prophet'	Run pip install prophet
Can't get attribute 'TextSummarizer'	Re-run src/summarizer.py to rebuild summarizer.pkl
No valid feedback column found	Rename column to feedback, review, or comments
plotly error	Run pip install plotly
Model not predicting	Ensure all .pkl files exist inside models/

👨‍💻 Author
Vishwajeet Singh
🎓 B.Tech CSE | 🧑‍💻 AI & MERN Developer | 📍 Mathura

🔗 LinkedIn | GitHub

⭐ Contributing
Contributions are welcome!

Fork this repository

Create your feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Added new feature")

Push to the branch (git push origin feature-name)

Open a Pull Request 🎉

📜 License
This project is licensed under the MIT License – feel free to use and modify.

💬 “AI that reads feedback, summarizes emotions, and visualizes insights — all in one click.”
yaml
Copy code

---

✅ This Markdown file is **ready to be pasted** into your GitHub repo as `README.md`.  
Would you like me to also generate a matching `requirements.txt` file for your repository?
