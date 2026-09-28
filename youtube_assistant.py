import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


load_dotenv()

# Render's Secret File location; use the local file when running on your computer.
transcript_path = Path("/etc/secrets/transcript.txt")
if not transcript_path.exists():
    transcript_path = Path(__file__).with_name("transcript.txt")

if not transcript_path.exists():
    raise FileNotFoundError(
        f"Transcript file not found at {transcript_path}. "
        "Add transcript.txt locally or as a Render Secret File."
    )

transcript = transcript_path.read_text(encoding="utf-8")

if not transcript.strip():
    raise ValueError("The transcript file is empty.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)
transcript_chunks = splitter.split_text(transcript)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(chunk) for chunk in transcript_chunks])


def find_relevant_sections(question: str, count: int = 4) -> str:
    scores = bm25.get_scores(tokenize(question))

    best_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:count]

    if not best_indices or scores[best_indices[0]] <= 0:
        return ""

    # Return the selected sections in their original video order.
    best_indices.sort()
    return "\n\n".join(transcript_chunks[index] for index in best_indices)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)


@tool
def answer_from_video(question: str) -> str:
    """Find relevant transcript sections and answer a question about the video."""
    relevant_sections = find_relevant_sections(question)

    if not relevant_sections:
        return "I couldn't find a matching section. Try asking with a specific topic or name."

    response = llm.invoke(
        "Answer using only these excerpts from the video transcript. "
        "If they don't answer the question, say so.\n\n"
        f"Transcript excerpts:\n{relevant_sections}\n\n"
        f"Question: {question}"
    )
    return response.content


agent = create_agent(
    model=llm,
    tools=[answer_from_video],
    system_prompt=(
        "You answer questions about the loaded video. "
        "Use the video tool to find answers."
    ),
)


def ask_video(question: str, history=None) -> str:
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content