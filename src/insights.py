import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from prophet import Prophet
from fpdf import FPDF

class InsightGenerator:
    def __init__(self):
        print("✅ InsightGenerator initialized.")

    # ---------- 1️⃣ Identify Recurring Issues ----------
    def extract_recurring_issues(self, df, text_column="clean_text", top_n=10):
        """Find top recurring words/phrases (complaints/issues)."""
        all_words = " ".join(df[text_column].astype(str)).split()
        common_words = Counter(all_words).most_common(top_n)
        recurring_issues = pd.DataFrame(common_words, columns=["Term", "Frequency"])
        return recurring_issues

    # ---------- 2️⃣ Predict Customer Satisfaction Trends ----------
    def forecast_satisfaction_trends(self, df, date_col="Date", rating_col="Rate"):
        """Predict satisfaction trend for next month using Prophet."""
        if date_col not in df.columns or rating_col not in df.columns:
            print("⚠️ Missing Date or Rate column. Forecast skipped.")
            return None
        
        forecast_df = df[[date_col, rating_col]].dropna().rename(columns={date_col: "ds", rating_col: "y"})
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot and save visualization
        plt.figure(figsize=(10, 5))
        model.plot(forecast)
        plt.title("Predicted Customer Satisfaction Trend")
        plt.savefig("reports/satisfaction_forecast.png")
        plt.close()

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # ---------- 3️⃣ Generate Visual Report ----------
    def generate_insight_report(self, df, output_path="reports/AI_insights_report.pdf"):
        """Generate PDF report with recurring issues and trend analysis."""
        os.makedirs("reports", exist_ok=True)

        # Create a word cloud
        text = " ".join(df["clean_text"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        wc.to_file("reports/wordcloud.png")

        # PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Customer Feedback Insights Report", ln=True, align="C")

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "\nThis report includes recurring issues, trend predictions, and visual analytics.")

        # Add WordCloud image
        pdf.image("reports/wordcloud.png", x=10, y=50, w=180)
        pdf.ln(110)

        # Add satisfaction trend chart (if generated)
        if os.path.exists("reports/satisfaction_forecast.png"):
            pdf.image("reports/satisfaction_forecast.png", x=10, y=170, w=180)
            pdf.ln(120)

        pdf.output(output_path)
        print(f"📊 Report saved to {output_path}")

# ---------- 4️⃣ Save as Pickle ----------
if __name__ == "__main__":
    df = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=60, freq="D"),
        "Rate": [3 + (i % 3) for i in range(60)],
        "clean_text": [
            "The delivery was late but product quality was good",
            "Packaging was poor, please improve service",
            "Excellent support team and fast delivery"
        ] * 20
    })

    insight_model = InsightGenerator()
    recurring = insight_model.extract_recurring_issues(df)
    print(recurring.head())

    forecast = insight_model.forecast_satisfaction_trends(df)
    if forecast is not None:
        print(forecast.tail())

    insight_model.generate_insight_report(df)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/insight_model.pkl", "wb") as f:
        pickle.dump(insight_model, f)
        print("\n✅ Saved InsightGenerator model to models/insight_model.pkl")
