import gradio as gr
import joblib
import pandas as pd
import datetime

# Load your model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction + logging function
def predict_news(news_input, news_file):
    # Prefer file input if present
    if news_file is not None:
        news = news_file.read().decode("utf-8")
    elif news_input.strip():
        news = news_input
    else:
        return "🚨 No news found. Gimme something to read, bro."

    try:
        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)[0]
        result = "🟢 Real News ✅" if prediction == 1 else "🔴 Fake News ❌"

        # Logging
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_snippet": news[:75] + "..." if len(news) > 75 else news,
            "prediction": result
        }
        df = pd.DataFrame([log_entry])
        df.to_csv("prediction_logs.csv", mode='a', header=False, index=False)

        return result
    except Exception as e:
        return f"💥 Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=predict_news,
    inputs=[
        gr.Textbox(lines=4, label="📝 Paste News Article"),
        gr.File(label="📁 Or Upload .txt File", file_types=[".txt"])
    ],
    outputs=gr.Text(label="🔍 Prediction"),
    title="🧠 Fake News Detector 2.0",
    description="Throw in your news (text or file) and let the AI snitch if it’s fake or real 🧐"
)

if __name__ == "__main__":
    iface.launch()
