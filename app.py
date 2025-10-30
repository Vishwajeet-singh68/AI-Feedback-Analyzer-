import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# ✅ Streamlit Config
# -------------------------------
st.set_page_config(page_title="AI Feedback Analysis System", layout="wide")
st.title("🤖 Intelligent Customer Feedback Analysis System")
st.write("Upload your feedback dataset to perform **Sentiment Analysis**, **Summarization**, and **Insights Generation**.")

# -------------------------------
# ✅ Utility: Safe Model Loader
# -------------------------------
def safe_load_model(path, model_name):
    """Load pickle model with full error handling"""
    try:
        if not os.path.exists(path):
            st.warning(f"⚠️ {model_name} not found at path: {path}")
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
            st.success(f"✅ {model_name} loaded successfully!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading {model_name}: {e}")
        return None

# -------------------------------
# ✅ Load All Models
# -------------------------------
sentiment_model = safe_load_model("models/sentiment_model.pkl", "Sentiment Classification Model")
summarizer_model = safe_load_model("models/summarizer.pkl", "Text Summarizer Model")
insight_model = safe_load_model("models/insight_model.pkl", "Predictive Insights Model")

# -------------------------------
# ✅ File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write("📊 **Preview of Uploaded Data:**")
        st.dataframe(df.head())

        # Detect feedback column
        possible_cols = ["feedback", "review", "comment", "text", "message", "customer_feedback"]
        feedback_col = None
        for col in df.columns:
            if col.lower() in possible_cols:
                feedback_col = col
                break

        if not feedback_col:
            st.error("❌ No valid feedback column found! Please ensure your dataset has one of: feedback, review, comment, text, or message.")
        else:
            st.success(f"✅ Feedback column detected: `{feedback_col}`")

            # ---------------------------------------
            # 🔹 SENTIMENT ANALYSIS
            # ---------------------------------------
            st.header("💬 Sentiment Analysis")

            if sentiment_model:
                try:
                    df["Predicted_Sentiment"] = df[feedback_col].apply(
                        lambda x: sentiment_model.predict([str(x)])[0]
                        if pd.notnull(x)
                        else "Unknown"
                    )
                    st.dataframe(df[[feedback_col, "Predicted_Sentiment"]].head(10))
                    st.subheader("📈 Sentiment Distribution")

                    sentiment_counts = df["Predicted_Sentiment"].value_counts()
                    fig, ax = plt.subplots()
                    sentiment_counts.plot(kind="bar", color=["green", "red", "gray"], ax=ax)
                    plt.title("Sentiment Distribution")
                    plt.xlabel("Sentiment Type")
                    plt.ylabel("Count")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"⚠️ Sentiment prediction failed: {e}")
            else:
                st.warning("⚠️ Sentiment model not loaded — skipping sentiment analysis.")

            # ---------------------------------------
            # 🔹 TEXT SUMMARIZATION
            # ---------------------------------------
            st.header("📝 Feedback Summarization")

            if summarizer_model:
                try:
                    short_summaries, long_summaries = summarizer_model.generate_summaries(
                        df[feedback_col].astype(str)
                    )
                    df["Short_Summary"] = short_summaries
                    df["Detailed_Summary"] = long_summaries
                    st.dataframe(df[[feedback_col, "Short_Summary", "Detailed_Summary"]].head(10))
                except Exception as e:
                    st.error(f"⚠️ Summarization failed: {e}")
            else:
                st.warning("⚠️ Summarizer model not loaded — skipping summarization.")

            # ---------------------------------------
            # 🔹 INSIGHT VISUALIZATION
            # ---------------------------------------
            st.header("📊 Predictive Insights & Trends")

            try:
                if "Predicted_Sentiment" in df.columns:
                    counts = df["Predicted_Sentiment"].value_counts()
                    st.bar_chart(counts)

                if insight_model and hasattr(insight_model, "generate_insights"):
                    st.write("🔮 Running Predictive Analysis...")
                    insight_model.generate_insights(df)
                    st.success("✅ Predictive insights generated successfully!")
                else:
                    st.info("ℹ️ No predictive model found. You can still visualize sentiments above.")
            except Exception as e:
                st.error(f"⚠️ Error generating insights: {e}")

            # ---------------------------------------
            # 🔹 DOWNLOAD PROCESSED DATA
            # ---------------------------------------
            st.header("💾 Download Results")
            try:
                output_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Processed Feedback CSV",
                    data=output_csv,
                    file_name="analyzed_feedback.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"⚠️ Error generating download: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
