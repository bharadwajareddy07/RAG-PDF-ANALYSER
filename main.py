import logging
import uuid
import os

from fastapi import FastAPI
from dotenv import load_dotenv

import inngest
import inngest.fast_api
from inngest.experimental import ai

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
    RAGChunkAndSrc,
)


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# Inngest Client
# ---------------------------------------------------------

inngest_client = inngest.Inngest(
    app_id="rag2",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# =========================================================
# 1. PDF INGESTION FUNCTION
# =========================================================

@inngest_client.create_function(
    fn_id="RAG : Inngest PDF",
    trigger=inngest.TriggerEvent(
        event="rag2/inngest_pdf"
    ),
)
async def rag_inngest_pdf(ctx: inngest.Context):

    # -----------------------------------------------------
    # Load and chunk PDF
    # -----------------------------------------------------

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:

        pdf_path = ctx.event.data["pdf_path"]

        source_id = ctx.event.data.get(
            "source_id",
            pdf_path
        )

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id
        )

    # -----------------------------------------------------
    # Embed and store in Qdrant
    # -----------------------------------------------------

    def _upsert(
        chunks_and_src: RAGChunkAndSrc
    ) -> RAGUpsertResult:

        chunks = chunks_and_src.chunks

        source_id = chunks_and_src.source_id

        # Create embeddings
        vecs = embed_texts(chunks)

        # Create unique IDs
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}"
                )
            )
            for i in range(len(chunks))
        ]

        # Create Qdrant payloads
        payloads = [
            {
                "source": source_id,
                "text": chunks[i]
            }
            for i in range(len(chunks))
        ]

        # Store in Qdrant
        QdrantStorage().upsert(
            ids,
            vecs,
            payloads
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    # -----------------------------------------------------
    # Step 1: Load and chunk
    # -----------------------------------------------------

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    # -----------------------------------------------------
    # Step 2: Embed and upsert
    # -----------------------------------------------------

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


# =========================================================
# 2. QUERY PDF FUNCTION
# =========================================================

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag2/query_pdf_ai"
    ),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    # -----------------------------------------------------
    # Search Qdrant
    # -----------------------------------------------------

    def _search(
        question: str,
        top_k: int = 5
    ) -> RAGSearchResult:

        # Create embedding for the user's question
        query_vec = embed_texts(
            [question]
        )[0]

        # Connect to Qdrant
        store = QdrantStorage()

        # Search for similar chunks
        found = store.search(
            query_vec,
            top_k
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    # -----------------------------------------------------
    # Get question from event
    # -----------------------------------------------------

    question = ctx.event.data["question"]

    top_k = int(
        ctx.event.data.get(
            "top_k",
            5
        )
    )

    # -----------------------------------------------------
    # Step 1: Embed and search
    # -----------------------------------------------------

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(
            question,
            top_k
        ),
        output_type=RAGSearchResult
    )

    # -----------------------------------------------------
    # Build context for the LLM
    # -----------------------------------------------------

    context_block = "\n\n".join(
        found.contexts
    )

    user_context = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # -----------------------------------------------------
    # Groq LLM Adapter
    # -----------------------------------------------------

    adapter = ai.openai.Adapter(
     auth_key=os.getenv("GROQ_API_KEY"),
     base_url="https://api.groq.com/openai/v1",
     model="openai/gpt-oss-120b"
     )

    # -----------------------------------------------------
    # Step 2: Generate answer using Groq
    # -----------------------------------------------------

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 512,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only "
                        "the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": user_context
                }
            ]
        }
    )

    # -----------------------------------------------------
    # Extract answer
    # -----------------------------------------------------

    answer = res[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ]

    # -----------------------------------------------------
    # Return final response
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI()


@app.get("/")
async def root():

    return {
        "message": "RAG2 API is running"
    }


# =========================================================
# INNGEST + FASTAPI
# =========================================================

inngest.fast_api.serve(
    app,
    inngest_client,
    [
        rag_inngest_pdf,
        rag_query_pdf_ai
    ],
)