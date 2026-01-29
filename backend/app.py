from utils.retrieve import build_en2th_prompt, extract_json, inference_mt

from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# =========================
# FastAPI schemas
# =========================
class EnglishRequest(BaseModel):
    wipo_id: int
    english: str

class TranslateResponse(BaseModel):
    translation: str
    

# ============
# API
# ============
app = FastAPI(title="En2Th Translation API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate_en2th", response_model=TranslateResponse)
def translate_en2th(req: EnglishRequest):
    chat_str = build_en2th_prompt(
        wipo_id=req.wipo_id,
        english=req.english,
    )

    # vLLM inference
    raw_output = inference_mt(chat_str)
    thai_cleaned = extract_json(raw_output, en2th=True)

    return TranslateResponse(
        translation=thai_cleaned,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
