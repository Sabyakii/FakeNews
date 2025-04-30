import os
import joblib
import gradio as gr
import pandas as pd

# Path to the model file
model_path = "E:\\fakenewsapp\\fake_news_model.joblib"

# Load the model
model = joblib.load(model_path)

# Load CSV file and get headlines
csv_file_path = 'E:\\fakenewsapp\\train.csv'  # Update with the correct path
df = pd.read_csv(csv_file_path)
headlines = df['title']  # Update column name if needed

# Prediction function
def predict_news(news):
    result = model.predict([news])[0]
    return f"🧠 Prediction: **{result.upper()}**"

# Gradio Interface
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=2, placeholder="Paste news headline here..."),
    outputs="markdown",
    title="📰 Fake News Detector",
    description="Enter a news headline. I’ll call it out if it’s fake. Built with ML. 🧠💣",
)

demo.launch()
