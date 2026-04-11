from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import spacy

# 1. Initialize FastAPI and Load NLP tools
app = FastAPI(title="Sentiment Analysis API")
nlp = spacy.load("en_core_web_sm")

# 2. Define the Model Architecture (Must match your training script)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.sigmoid(self.fc(cat))

# 3. Load the pre-trained weights
# Note: Ensure vocab_size matches your saved vocabulary mapping
model = SentimentLSTM(vocab_size=10000, embed_dim=100, hidden_dim=256)
model.load_state_dict(torch.load("models/sentiment_model.pth", map_location=torch.device('cpu')))
model.eval()

class RequestData(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(data: RequestData):
    # Preprocessing: Tokenization & Numericalization
    tokens = [token.text.lower() for token in nlp(data.text)]
    # (Mapping logic to numbers would go here based on your training vocab)
    # For demo: dummy tensor representing a processed sentence
    dummy_input = torch.randint(0, 10000, (1, 50)) 
    
    with torch.no_grad():
        prediction = model(dummy_input).item()
    
    sentiment = "positive" if prediction > 0.5 else "negative"
    return {"text": data.text, "sentiment": sentiment, "confidence": round(prediction, 4)}