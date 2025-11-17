import os
import json
from typing import Optional, List, Dict, Any
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import create_document, get_documents

# Environment placeholders (replace via real env vars later)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "YOUR_PERPLEXITY_KEY_HERE")
OPENAI_REALTIME_API_KEY = os.getenv("OPENAI_REALTIME_API_KEY", "YOUR_CHATGPT_VOICE_KEY_HERE")

app = FastAPI(title="PokéValue UK Voice Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PriceRequest(BaseModel):
    card: str
    user_id: Optional[str] = None

class PriceResult(BaseModel):
    card_name: str
    last_sold_price: Optional[float] = None
    last_sold_date: Optional[str] = None
    median_sold_price: Optional[float] = None
    average_sold_price: Optional[float] = None
    sample_size: int = 0
    confidence_score: str = "low"
    top_listing_url: Optional[str] = None
    raw_samples: Optional[List[Dict[str, Any]]] = None

@app.get("/")
def root():
    return {"service": "PokeValue UK Backend", "status": "ok"}

@app.post("/api/price", response_model=PriceResult)
def fetch_price(body: PriceRequest):
    if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == "YOUR_PERPLEXITY_KEY_HERE":
        raise HTTPException(status_code=500, detail="Perplexity API key not configured")

    prompt = (
        f"Find the 10 most recent SOLD eBay UK listings for {body.card}, "
        "extract: sold price (GBP), sold date (ISO), condition, and link. "
        "Return ONLY valid JSON with keys: samples (array of {price, date, condition, url}), "
        "last_sold_price, last_sold_date, median_sold_price, average_sold_price, sample_size, confidence_score, top_listing_url."
    )

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are a UK price extraction engine. Always use eBay UK SOLD listings only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=40,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Perplexity error: {resp.text[:200]}")
        data = resp.json()
        # The API returns completions; extract the text content
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            # Some responses may be under choices[0].text
            content = data.get("choices", [{}])[0].get("text", "")
        # Ensure we have JSON string in content
        # Try to find JSON block if wrapped in markdown
        json_str = content
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        parsed = {}
        try:
            parsed = json.loads(json_str)
        except Exception:
            parsed = {}
        samples = parsed.get("samples", [])
        # Compute derived metrics if missing
        prices = [s.get("price") for s in samples if isinstance(s.get("price"), (int, float))]
        last_price = parsed.get("last_sold_price") or (prices[0] if prices else None)
        last_date = parsed.get("last_sold_date") or (samples[0].get("date") if samples else None)
        median_price = parsed.get("median_sold_price")
        avg_price = parsed.get("average_sold_price")
        if prices:
            if median_price is None:
                sp = sorted(prices)
                mid = len(sp)//2
                median_price = (sp[mid] if len(sp)%2==1 else (sp[mid-1]+sp[mid])/2)
            if avg_price is None:
                avg_price = sum(prices)/len(prices)
        sample_size = parsed.get("sample_size") or len(samples)
        confidence = parsed.get("confidence_score") or ("high" if sample_size>=8 else ("medium" if sample_size>=4 else "low"))
        top_url = parsed.get("top_listing_url") or (samples[0].get("url") if samples else None)

        if not samples and (median_price is None or avg_price is None):
            return PriceResult(
                card_name=body.card,
                last_sold_price=None,
                last_sold_date=None,
                median_sold_price=None,
                average_sold_price=None,
                sample_size=0,
                confidence_score="low",
                top_listing_url=None,
                raw_samples=[],
            )

        result = PriceResult(
            card_name=body.card,
            last_sold_price=(round(last_price,2) if isinstance(last_price,(int,float)) else None),
            last_sold_date=last_date,
            median_sold_price=(round(median_price,2) if isinstance(median_price,(int,float)) else None),
            average_sold_price=(round(avg_price,2) if isinstance(avg_price,(int,float)) else None),
            sample_size=int(sample_size or 0),
            confidence_score=str(confidence),
            top_listing_url=top_url,
            raw_samples=samples,
        )
        # Persist search for history
        try:
          if body.user_id:
            from schemas import Search
            doc = Search(
              user_id=body.user_id,
              card_name=result.card_name,
              last_sold_price=result.last_sold_price,
              last_sold_date=result.last_sold_date,
              median=result.median_sold_price,
              average=result.average_sold_price,
            )
            create_document("search", doc)
        except Exception:
          pass
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price: {str(e)[:200]}")

@app.get("/api/history")
def history(user_id: str = Query(...)):
    try:
        docs = get_documents("search", {"user_id": user_id}, limit=50)
        mapped = [
            {
                "card_name": d.get("card_name"),
                "last_sold_price": d.get("last_sold_price"),
                "median": d.get("median"),
                "average": d.get("average"),
                "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
            }
            for d in docs
        ]
        return mapped
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)[:200]}")

@app.get("/api/realtime/token")
def create_realtime_session():
    """
    Create short-lived OpenAI Realtime session. Also instructs a lightweight protocol:
    - When the user asks for a card price, send a data channel JSON message: {"type":"price_query", "card":"<name>"}
    - The client will call the backend /api/price and reply with {"type":"price_result", "payload": <PriceResult>}
    - After receiving price_result, speak a concise summary and ALWAYS say: "Based on eBay UK sold listings."
    - Use friendly Pokédex/Professor Oak tone and add an optional fun line (e.g., "Pikachu says…")
    """
    if not OPENAI_REALTIME_API_KEY or OPENAI_REALTIME_API_KEY == "YOUR_CHATGPT_VOICE_KEY_HERE":
        raise HTTPException(status_code=500, detail="OpenAI Realtime API key not configured")

    try:
        resp = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_REALTIME_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-realtime-preview-2024-12-17",
                "voice": "verse",
                "modalities": ["audio", "text"],
                "instructions": (
                    "You are PokéValue UK, a friendly Pokémon-themed UK pricing expert. "
                    "Conversation is full-duplex voice. Maintain turn taking naturally. "
                    "When the user asks a price question, emit a data channel message JSON {type:'price_query', card:'<name>'}. "
                    "Wait for {type:'price_result'} to arrive, then summarize with exact GBP numbers and say: 'Based on eBay UK sold listings.' "
                    "If results are missing or sample is small, say confidence is low due to limited UK sales. Never use US or active listings."
                )
            },
            timeout=20,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"OpenAI session error: {resp.text[:200]}")
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating realtime session: {str(e)[:200]}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
