"""
FastAPI service that scans a JSON object (uploaded file or raw body) for
suspicious (XSS / SQL injection) content, field by field, using the
trained CharCNNBiLSTM model.

WHY FIELD-BY-FIELD:
Testing earlier showed the model scores nonsense on a whole raw HTTP
request or a whole JSON blob at once (headers/JWTs/JSON punctuation
confuse it). It performs reliably on individual short string values --
the way it was trained. So this API recursively walks the JSON,
extracts every string leaf value, scores each one independently, and
aggregates the results into a per-field + overall verdict.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then either:
    - POST a JSON body directly to /scan
    - POST a .json file to /scan-file
"""

import json
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Union

# ---------------------------------------------------------------------
# Model definition (must match train.py exactly)
# ---------------------------------------------------------------------
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
        e = self.embedding(x).permute(0, 2, 1)
        c = self.relu(self.conv1(e)); c = self.pool(c)
        c = self.relu(self.conv2(c)); c = self.pool(c)
        c = c.permute(0, 2, 1)
        out, (h, _) = self.lstm(c)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        h_cat = self.dropout(h_cat)
        z = self.relu(self.fc1(h_cat))
        logit = self.fc2(z).squeeze(1)
        return logit

# ---------------------------------------------------------------------
# Load trained model + vocab once at startup
# ---------------------------------------------------------------------
MODEL_PATH = "best_model.pt"
THRESHOLD = 0.5  # tune this: lower = catch more attacks, more false positives

device = torch.device("cpu")
ckpt = torch.load(MODEL_PATH, map_location=device)
char2idx = ckpt["char2idx"]
max_len = ckpt["max_len"]
vocab_size = ckpt["vocab_size"]

model = CharCNNBiLSTM(vocab_size)
model.load_state_dict(ckpt["model_state"])
model.eval()


def encode(text: str, max_len: int = max_len):
    ids = [char2idx.get(c, 1) for c in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


def score_string(text: str) -> float:
    """Return P(suspicious) for a single string."""
    if not text.strip():
        return 0.0
    x = torch.tensor([encode(text)], dtype=torch.long)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob


def walk_json(obj: Any, path: str = "$") -> List[Dict[str, Any]]:
    """
    Recursively walk a parsed JSON object/list and return a flat list of
    {path, value} for every string leaf found. Non-string leaves
    (numbers, bools, null) are skipped -- the model only makes sense on text.
    """
    leaves = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves.extend(walk_json(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves.extend(walk_json(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        leaves.append({"path": path, "value": obj})
    # ints/floats/bools/None are intentionally skipped
    return leaves


def scan_json_object(obj: Any) -> Dict[str, Any]:
    leaves = walk_json(obj)
    field_results = []
    for leaf in leaves:
        prob = score_string(leaf["value"])
        field_results.append({
            "path": leaf["path"],
            "value": leaf["value"][:200],  # truncate long values in the response
            "probability": round(prob, 4),
            "suspicious": prob >= THRESHOLD,
        })

    flagged = [f for f in field_results if f["suspicious"]]
    max_prob = max([f["probability"] for f in field_results], default=0.0)

    return {
        "overall_verdict": "SUSPICIOUS" if flagged else "benign",
        "fields_scanned": len(field_results),
        "fields_flagged": len(flagged),
        "highest_risk_score": max_prob,
        "flagged_fields": flagged,
        "all_fields": field_results,
    }


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="XSS / SQLi JSON Scanner",
    description="Scores each string field in a JSON payload for XSS/SQL-injection risk.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "POST a JSON body to /scan, or upload a .json file to /scan-file",
        "threshold": THRESHOLD,
    }


@app.post("/scan")
def scan_json_body(payload: Union[Dict[str, Any], List[Any]]):
    """Scan a raw JSON object/array sent directly as the request body."""
    return scan_json_object(payload)


@app.post("/scan-file")
async def scan_json_file(file: UploadFile = File(...)):
    """Scan an uploaded .json file."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a .json file")
    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    return scan_json_object(data)


@app.post("/scan-text")
def scan_single_string(item: Dict[str, str]):
    """
    Convenience endpoint: scan a single raw string.
    Body: {"text": "..."}
    """
    text = item.get("text", "")
    prob = score_string(text)
    return {
        "text": text[:200],
        "probability": round(prob, 4),
        "suspicious": prob >= THRESHOLD,
    }