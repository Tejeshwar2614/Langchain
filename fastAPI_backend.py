from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import youtube_assistant as yap
from fastapi import FastAPI, HTTPException
from groq import RateLimitError


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)

@app.post("/chat")
def chat(request: ChatRequest):
    answer = yap.ask_video(request.question, request.history)
    return {"answer": answer}

# Keep your other imports and setup.

class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = yap.ask_video(request.question, request.history)
        return {"answer": answer}
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="Groq's token limit was reached. Wait for it to reset, then try again.",
        ) from exc