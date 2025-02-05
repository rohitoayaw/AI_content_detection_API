from fastapi import FastAPI, HTTPException
import transformers
import torch
import torch.nn.functional as F
from typing import List
from cachetools import TTLCache
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

app = FastAPI()

device = "cpu"
model_name = "TrustSafeAI/RADAR-Vicuna-7B"
detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
detector.eval()
detector.to(device)

# Cache setup
cache = TTLCache(maxsize=1000, ttl=300)

@app.on_event("startup")
def load_model():
    
    pass

def predict(text_input: List[str]):
    with torch.no_grad():
        inputs = tokenizer(text_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp().tolist()
    return output_probs

@app.post("/predict/")
async def get_predictions(text_input: List[str]):
    cache_key = tuple(text_input)  # Cache key is the input texts tuple
    if cache_key in cache:
        return {"probabilities": cache[cache_key]}
    
    probabilities = predict(text_input)
    cache[cache_key] = probabilities
    return {"probabilities": probabilities}
