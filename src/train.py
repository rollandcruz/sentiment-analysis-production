import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import spacy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# 1. Configuration & Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
EPOCHS = 5
MODEL_SAVE_PATH = "models/sentiment_model.pth"

# Load spaCy for tokenization
nlp = spacy.load("en_core_web_sm")

# 2. Simple Dataset Class
class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, vocab):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # Numericalize tokens using the vocab
        review = [self.vocab.get(token.text.lower(), 1) for token in nlp(self.reviews[idx])]
        # Simple padding/truncating to length 100
        review = review[:100] + [0] * (100 - len(review)) 
        return torch.tensor(review), torch.tensor(self.labels[idx], dtype=torch.float32)

# 3. Model Architecture (Matches your API script)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.sigmoid(self.fc(cat))

# 4. Main Training Loop
def train():
    # Create directory for model if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load your CSV data (Ensure you have downloaded IMDB Dataset.csv)
    print("Loading data...")
    df = pd.read_csv('data/IMDB Dataset.csv')
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Build simple Vocabulary (In production, use a more robust Vocab object)
    print("Building vocabulary...")
    all_text = " ".join(df['review'].iloc[:2000].tolist()) # Sample for vocab
    tokens = [token.text.lower() for token in nlp(all_text)]
    vocab = {word: i+2 for i, (word, _) in enumerate(Counter(tokens).most_common(MAX_VOCAB_SIZE))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    # Split Data
    train_df, val_df = train_test_split(df, test_size=0.2)
    
    train_ds = IMDBDataset(train_df['review'].values, train_df['label'].values, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = SentimentLSTM(MAX_VOCAB_SIZE + 2, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"Starting training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()