import os
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class TextSummarizer:
    """
    Handles feedback text summarization using a Transformer model.
    Automatically downloads and caches the model locally for reuse.
    """

    def __init__(self, model_name="facebook/bart-large-cnn", model_dir="models/summarizer"):
        self.model_name = model_name
        self.model_dir = model_dir

        # Load or download model
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"✅ Loading existing summarization model from {model_dir} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        else:
            print(f"⬇️ Downloading new summarization model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._save_model()

        # Initialize summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )

    def _save_model(self):
        """Save model and tokenizer locally for offline reuse"""
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"💾 Saving summarization model to {self.model_dir} ...")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def generate_summaries(self, texts, short_len=30, long_len=100):
        """
        Generate short and detailed summaries for multiple feedback texts.
        Returns two lists: short_summaries and detailed_summaries.
        """
        short_summaries, detailed_summaries = [], []

        for text in texts:
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                msg = "Too short to summarize"
                short_summaries.append(msg)
                detailed_summaries.append(msg)
                continue

            try:
                short_sum = self.summarizer(
                    text, max_length=short_len, min_length=10, do_sample=False
                )[0]['summary_text']

                long_sum = self.summarizer(
                    text, max_length=long_len, min_length=40, do_sample=False
                )[0]['summary_text']

            except Exception as e:
                print(f"⚠️ Summarization failed for one text: {e}")
                short_sum, long_sum = "Error generating summary", "Error generating summary"

            short_summaries.append(short_sum)
            detailed_summaries.append(long_sum)

        return short_summaries, detailed_summaries


if __name__ == "__main__":
    # Demo run + save summarizer.pkl
    demo_texts = [
        "The product quality was excellent but delivery was late. I hope next time it arrives sooner.",
        "Customer service was poor and the item was damaged."
    ]

    summarizer = TextSummarizer()
    short_summaries, long_summaries = summarizer.generate_summaries(demo_texts)

    for i, text in enumerate(demo_texts):
        print(f"\n📝 Original: {text}")
        print(f"➡️ Short Summary: {short_summaries[i]}")
        print(f"➡️ Detailed Summary: {long_summaries[i]}")

    summarizer_pkl_path = os.path.join("models", "summarizer.pkl")
    os.makedirs("models", exist_ok=True)

    with open(summarizer_pkl_path, "wb") as f:
        pickle.dump(summarizer, f)

    print(f"\n✅ Summarizer pipeline saved successfully at {summarizer_pkl_path}")
