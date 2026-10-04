from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


load_dotenv()


# Groq client - use this for your LLM/chat generation
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# Local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# all-MiniLM-L6-v2 produces 384-dimensional vectors
EMBED_DIM = 384


# PDF chunking
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=200
)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    text = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for t in text:
        chunks.extend(splitter.split_text(t))

    return chunks


def embed_texts(text: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        text,
        convert_to_numpy=True
    )

    return embeddings.tolist()