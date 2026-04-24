import whisper
from transformers import pipeline

wm = whisper.load_model("base")

summ = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def speech_summary(path):
    text = wm.transcribe(path)["text"]
    chunks = [text[i:i+1000] for i in range(0,len(text),1000)]
    return " ".join([summ(c)[0]["summary_text"] for c in chunks])