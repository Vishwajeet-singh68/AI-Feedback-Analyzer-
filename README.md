🧠 AI Feedback Analysis Web App

An AI-powered web application that analyzes customer feedback, predicts sentiment, summarizes text, and generates insights — built using Python, Streamlit, and Hugging Face Transformers.

🚀 Features

🗂️ Upload feedback data in .csv format

😃 Perform Sentiment Analysis using DistilBERT

✍️ Generate Summaries of feedback

📊 Visualize Insights (positive, neutral, negative distribution)

🌐 Easy-to-use Streamlit Web Interface

🧩 Tech Stack
Component	Technology
Frontend UI	Streamlit
Backend	Python
ML Model	Hugging Face Transformers (DistilBERT)
Visualization	Plotly, Matplotlib
File Handling	Pandas, CSV
📁 Project Structure
AI-Feedback-Analyzer/
│
├── app.py                          # Streamlit main application
├── models/
│   ├── sentiment_model.pkl         # Trained sentiment model (auto-created)
│   ├── summarizer.pkl              # Summarizer model (auto-created)
│   └── insights_model.pkl          # Insights model (auto-created)
│
├── src/
│   ├── sentiment_model.py          # Builds and saves sentiment model
│   ├── summarizer.py               # Text summarizer class
│   ├── insights.py                 # Insight generator using Prophet
│
├── requirements.txt                # Required dependencies
└── README.md                       # Setup guide (this file)

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/AI-Feedback-Analyzer.git
cd AI-Feedback-Analyzer

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows
# or
source venv/bin/activate  # For macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Model Setup Files

These will create .pkl models in the models/ directory.

python src/sentiment_model.py
python src/summarizer.py
python src/insights.py

🧾 CSV File Format

Your feedback file should be a .csv with at least one of these columns:

feedback

review

comments

Example:

feedback
The product quality is amazing!
Worst experience ever!
It's okay, nothing special.
▶️ Run the Streamlit App

Once models are created, launch the app:

streamlit run app.py

📊 Output Preview

✅ Upload CSV →
✅ App performs Sentiment Analysis →
✅ Shows Positive/Negative charts →
✅ Displays text summaries and trend insights

🧠 Example Workflow

Upload your feedback.csv

App analyzes sentiments

App generates summaries

App visualizes overall insights

🧰 Requirements (requirements.txt)
streamlit
torch
transformers
pandas
plotly
matplotlib
prophet
scikit-learn

💡 Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named 'prophet'	Run pip install prophet
Can't get attribute 'TextSummarizer'	Ensure the same class name is used while saving/loading model
CSV upload error	Ensure column name is feedback or review
Importing plotly failed	Run pip install plotly
