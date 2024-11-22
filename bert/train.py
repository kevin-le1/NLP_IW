from modal import App, Image, Volume
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split as tts
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict
app = App()

# Define the image with required dependencies
image = (
    Image.debian_slim()
    .apt_install(
    "libgl1-mesa-glx",          # OpenGL support
    "libglib2.0-0",             # GLib networking and filesystem utilities
    "libsm6",                   # X Session Management library (often needed for OpenCV)
    "libxrender1",              # Rendering extension to X11 (for GUI applications)
    "libxext6",                 # Miscellaneous X11 utilities
    "ffmpeg",                   # For handling video and audio files
    "libopencv-dev",            # OpenCV development libraries
    "libglib2.0-dev",           # GLib development libraries for compiling dependent programs
)
    .pip_install(
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "scikit-learn",
        "tqdm",
    )
)

# Create a volume to store the dataset and model checkpoints
volume = Volume.from_name("phishing-persisted-volume", create_if_missing=True)

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
EPOCHS = 10
MAX_LEN = 512
BATCH_SIZE = 16

# Dataset class
class PhishingCollection(Dataset):
    def __init__(self, phishing, msgs, tokenizer, max_len):
        self.msgs = msgs
        self.phishing = phishing
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, i):
        msg = str(self.msgs[i])
        phishing = self.phishing[i]

        encoding = self.tokenizer.encode_plus(
            msg,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'msg': msg,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'phishing': torch.tensor(phishing, dtype=torch.long)
        }

# Function to create data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PhishingCollection(
        phishing=df['Email Type'].to_numpy(),
        msgs=df['Email Text'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)

# Function to load and preprocess dataset
@app.function(volumes={"/root/dataset": volume}, image=image, timeout=600)
def load_and_preprocess_data():
    dataset = pd.read_csv('/root/dataset/Phishing_Email.csv')[['Email Text', 'Email Type']]
    print(dataset.head(5))
    
    dataset['Email Type'] = dataset['Email Type'].str.lower().str.strip().apply(lambda x: 1 if x == 'phishing email' else 0)
    df_train, df_test = tts(dataset, test_size=0.2, random_state=42)
    df_val, df_test = tts(df_test, test_size=0.5, random_state=42)

    df_train.to_pickle("/root/dataset/df_train.pkl")
    df_val.to_pickle("/root/dataset/df_val.pkl")
    df_test.to_pickle("/root/dataset/df_test.pkl")
    
    volume.commit()
    print("Data split and saved to volume.")

# Model class
class PhishingClassifier(nn.Module):
    def __init__(self, n_classes):
        super(PhishingClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        return self.out(output)

# Function to train the model
@app.function(volumes={"/root/dataset": volume}, image=image, gpu="A100", timeout=86400)
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data from volume
    df_train = pd.read_pickle("/root/dataset/df_train.pkl")
    df_val = pd.read_pickle("/root/dataset/df_val.pkl")

    # Create data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    # Initialize model, optimizer, scheduler, and loss function
    model = PhishingClassifier(n_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training loop
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # Training phase
        model.train()
        train_losses = []
        correct_predictions = 0

        for d in tqdm(train_data_loader, desc="Training", leave=False):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['phishing'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            train_losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(df_train)
        train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in tqdm(val_data_loader, desc="Validation", leave=False):
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                targets = d['phishing'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, targets)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == targets)
                val_losses.append(loss.item())

        val_acc = correct_predictions.double() / len(df_val)
        val_loss = np.mean(val_losses)

        print(f'Train loss {train_loss}, Train accuracy {train_acc}')
        print(f'Val loss {val_loss}, Val accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), '/root/dataset/best_model_state.bin')
            best_accuracy = val_acc
            volume.commit()

        print(history)

    print("Training complete.")

# Main entry point
@app.local_entrypoint()
def main():
    # Load the dataset locally
    # load_and_preprocess_data.remote()
    train_model.remote()