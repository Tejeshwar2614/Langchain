# from langchain_community.document_loaders import YoutubeLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain.tools import tool

# load_dotenv()

# # checking whether transcript available for this or not

# video_url = "https://youtu.be/x63HCoDfAhQ"

# loader = YoutubeLoader.from_youtube_url(
#     video_url,
#     add_video_info=False,
#     language=["en"],
# )

# documents = loader.load()
# Transcript = documents[0].page_content

# # print("Documents loaded:", len(documents))
# # print("Video info:", documents[0].metadata)
# # print("Transcript preview:")
# # print(documents[0].page_content[:1000])

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 6000,
#     chunk_overlap = 300,
# )

# chunks = splitter.split_text(Transcript)

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0.1,
# )

# chunk_summaries = []
            
# for number, chunk in enumerate(chunks, start=1):
#     response = llm.invoke(
#         "Summarize this section of a horror podcast. Keep the main stories, "
#         "people, and events. Don't add details that aren't in the transcript.\n\n"
#         f"Section {number}:\n{chunk}"
#     )
#     chunk_summaries.append(response.content)

# # Combine the section summaries into one overall summary.
# final_response = llm.invoke(
#     "Combine these section summaries into a clear overall summary of the podcast. "
#     "Mention recurring stories or themes, and don't invent details.\n\n"
#     + "\n\n".join(chunk_summaries)
# )

# # print(final_response.content[:1000])

# def ask_video(question: str, history=None) -> str:
#     messages = list(history or [])
#     messages.append({"role": "user", "content": question})

#     result = agent.invoke({"messages": messages})
#     return result["messages"][-1].content

# # print(ask_video("What is the main message of the podcast?"))

# @tool
# def answer_from_video(question: str) -> str:
#     """Answer a question using the loaded video's transcript."""
#     response = llm.invoke(
#         "Answer using only this transcript. If the answer isn't there, say so.\n\n"
#         f"Transcript:\n{Transcript}\n\n"
#         f"Question: {question}"
        
#     )
#     return response.content


# agent = create_agent(
#     model=llm,
#     tools=[answer_from_video],
#     system_prompt="You answer questions about the loaded video. Use the video tool to find answers.",
# )

# result = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "What is the podcast's main message?"}
#     ]
# })


from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

load_dotenv()

video_url = "https://youtu.be/x63HCoDfAhQ"

loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,
    language=["en"],
)
documents = loader.load()

if not documents or not documents[0].page_content.strip():
    raise ValueError("No English transcript was available for this video.")

transcript = documents[0].page_content
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

    # Put the selected sections back in video order.
    best_indices.sort()
    return "\n\n".join(transcript_chunks[index] for index in best_indices)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)


# @tool
# def answer_from_video(question: str) -> str:
#     """Answer a question using the loaded video's transcript."""
#     response = llm.invoke(
#         "Answer using only this transcript. If the answer isn't there, say so.\n\n"
#         f"Transcript:\n{transcript}\n\n"
#         f"Question: {question}"
#     )
#     return response.content
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

