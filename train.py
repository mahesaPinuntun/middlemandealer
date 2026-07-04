import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json, time, sys, datetime, os

# =====================================================================
# TEE LOGGER — duplicates every print() to console AND a .txt file
# =====================================================================
class Tee:
    """Writes everything to both the terminal and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = f'train_output_{timestamp}.txt'
tee = Tee(LOG_PATH)
sys.stdout = tee
# from this point on, every print() call goes to BOTH the console and LOG_PATH

print("=" * 70)
print("XSS / SQL INJECTION DETECTOR — TRAINING RUN LOG")
print(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

df = pd.read_csv('combined.csv')
df['text'] = df['text'].astype(str)
print('dataset shape:', df.shape)
print('class balance:\n', df['Label'].value_counts().to_string())

# ---- Char vocabulary ----
MAX_LEN = 200
chars = sorted(list(set(''.join(df['text'].tolist()))))
# reserve 0 = PAD, 1 = UNK
char2idx = {c: i + 2 for i, c in enumerate(chars)}
vocab_size = len(char2idx) + 2
print('vocab size:', vocab_size)
print('max sequence length:', MAX_LEN)

def encode(text, max_len=MAX_LEN):
    ids = [char2idx.get(c, 1) for c in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids

X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'].values, df['Label'].values, test_size=0.2, random_state=42, stratify=df['Label'].values)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print('train/val/test sizes:', len(X_train), len(X_val), len(X_test))

class CharDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx]), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

train_ds = CharDataset(X_train, y_train)
val_ds = CharDataset(X_val, y_val)
test_ds = CharDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)
test_dl = DataLoader(test_ds, batch_size=256)

class CharCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, cnn_channels=64, lstm_hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(emb_dim, cnn_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e = self.embedding(x)              # (B, L, E)
        e = e.permute(0, 2, 1)              # (B, E, L)
        c = self.relu(self.conv1(e))
        c = self.pool(c)
        c = self.relu(self.conv2(c))
        c = self.pool(c)
        c = c.permute(0, 2, 1)              # (B, L', C)
        out, (h, _) = self.lstm(c)
        h_cat = torch.cat([h[0], h[1]], dim=1)  # bidirectional final states
        h_cat = self.dropout(h_cat)
        z = self.relu(self.fc1(h_cat))
        z = self.dropout(z)
        logit = self.fc2(z).squeeze(1)
        return logit

model = CharCNNBiLSTM(vocab_size).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print('trainable params:', n_params)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print('optimizer: Adam, lr=1e-3')
print('loss function: BCEWithLogitsLoss')
print('batch size: 128 (train), 256 (eval)')

def evaluate(dl):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            all_logits.append(logit.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == y_true).float().mean().item()
    loss = criterion(logits, y_true).item()
    return loss, acc, probs.numpy(), y_true.numpy()

train_eval_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

EPOCHS = 6
best_val_acc = 0
history = []
print(f'\nStarting training for {EPOCHS} epochs...')
print("-" * 70)
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    epoch_t0 = time.time()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    epoch_time = time.time() - epoch_t0
    train_loss = total_loss / len(train_ds)
    train_eval_loss, train_acc, _, _ = evaluate(train_eval_dl)
    val_loss, val_acc, _, _ = evaluate(val_dl)
    loss_gap = val_loss - train_eval_loss
    acc_gap = train_acc - val_acc
    history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                     'val_loss': val_loss, 'val_acc': val_acc,
                     'loss_gap': loss_gap, 'acc_gap': acc_gap, 'epoch_time_sec': epoch_time})
    print(f'Epoch {epoch}/{EPOCHS} - train_loss {train_loss:.4f} - train_acc {train_acc:.4f} '
          f'- val_loss {val_loss:.4f} - val_acc {val_acc:.4f} '
          f'- loss_gap {loss_gap:+.4f} - acc_gap {acc_gap:+.4f} - time {epoch_time:.1f}s')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state': model.state_dict(),
                    'char2idx': char2idx,
                    'max_len': MAX_LEN,
                    'vocab_size': vocab_size}, 'best_model.pt')
        print(f'  -> new best model saved (val_acc={val_acc:.4f})')

total_train_time = time.time() - t0
print("-" * 70)
print('total training time (s):', total_train_time)
print('total training time (min):', round(total_train_time / 60, 2))

# ---- final test evaluation using best checkpoint ----
ckpt = torch.load('best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
test_loss, test_acc, probs, y_true = evaluate(test_dl)
preds = (probs >= 0.5).astype(int)
print('\n' + "=" * 70)
print('FINAL TEST SET RESULTS (best checkpoint)')
print("=" * 70)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
print()
print(classification_report(y_true, preds, target_names=['benign', 'suspicious']))
cm = confusion_matrix(y_true, preds)
print('confusion matrix:')
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f'\nTrue Negatives  : {tn}')
print(f'False Positives : {fp}')
print(f'False Negatives : {fn}')
print(f'True Positives  : {tp}')
print(f'False Positive Rate : {fp/(fp+tn):.4f}')
print(f'False Negative Rate : {fn/(fn+tp):.4f}')
auc = roc_auc_score(y_true, probs)
print('\nROC-AUC:', auc)

report = classification_report(y_true, preds, target_names=['benign', 'suspicious'], output_dict=True)
results = {
    'run_timestamp': timestamp,
    'device': str(device),
    'dataset_shape': list(df.shape),
    'vocab_size': vocab_size,
    'max_len': MAX_LEN,
    'trainable_params': n_params,
    'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
    'epochs': EPOCHS,
    'total_train_time_sec': total_train_time,
    'test_loss': test_loss,
    'test_acc': test_acc,
    'roc_auc': float(auc),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'false_positive_rate': float(fp / (fp + tn)),
    'false_negative_rate': float(fn / (fn + tp)),
    'report': report,
    'history': history,
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + "=" * 70)
print('Saved best_model.pt, results.json')
print(f'Full console log saved to: {LOG_PATH}')
print("=" * 70)

# restore stdout and close the log file cleanly
sys.stdout = tee.terminal
tee.close()
print(f'\nDone. Full training log written to: {os.path.abspath(LOG_PATH)}')