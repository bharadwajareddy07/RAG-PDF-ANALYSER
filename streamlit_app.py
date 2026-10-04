import asyncio
from pathlib import Path
import time
import os
import requests

import streamlit as st
import inngest
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered",
)


# ---------------------------------------------------------
# Inngest client
# ---------------------------------------------------------

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag2",
        is_production=False,
    )


# ---------------------------------------------------------
# Save uploaded PDF
# ---------------------------------------------------------

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)

    return file_path


# ---------------------------------------------------------
# Send PDF ingestion event
# ---------------------------------------------------------

async def send_rag_ingest_event(pdf_path: Path):
    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag2/inngest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

    return result


# ---------------------------------------------------------
# Upload UI
# ---------------------------------------------------------

st.title("Upload a PDF to Ingest")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded is not None:

    with st.spinner("Uploading and triggering ingestion..."):

        path = save_uploaded_pdf(uploaded)

        asyncio.run(
            send_rag_ingest_event(path)
        )

        time.sleep(0.3)

    st.success(f"Triggered ingestion for: {path.name}")

    st.caption(
        "You can upload another PDF if you like."
    )


st.divider()

st.title("Ask a question about your PDFs")


# ---------------------------------------------------------
# Send RAG query event
# ---------------------------------------------------------

async def send_rag_query_event(
    question: str,
    top_k: int,
) -> str:

    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag2/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    # Inngest returns event IDs
    return result[0]


# ---------------------------------------------------------
# Inngest local API
# ---------------------------------------------------------

def _inngest_api_base() -> str:

    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1",
    )


# ---------------------------------------------------------
# Get runs for an event
# ---------------------------------------------------------

def fetch_runs(event_id: str) -> list[dict]:

    url = (
        f"{_inngest_api_base()}"
        f"/events/{event_id}/runs"
    )

    resp = requests.get(
        url,
        timeout=10,
    )

    resp.raise_for_status()

    data = resp.json()

    return data.get("data", [])


# ---------------------------------------------------------
# Wait for Inngest function output
# ---------------------------------------------------------

def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
) -> dict:

    start = time.time()

    last_status = None

    while True:

        runs = fetch_runs(event_id)

        if runs:

            run = runs[0]

            status = run.get("status")

            last_status = status or last_status

            # Successful run
            if status in (
                "Completed",
                "Succeeded",
                "Success",
                "Finished",
            ):
                return run.get("output") or {}

            # Failed run
            if status in (
                "Failed",
                "Cancelled",
            ):
                raise RuntimeError(
                    f"Function run {status}"
                )

        # Timeout
        if time.time() - start > timeout_s:

            raise TimeoutError(
                "Timed out waiting for run output "
                f"(last status: {last_status})"
            )

        time.sleep(poll_interval_s)


# ---------------------------------------------------------
# Query UI
# ---------------------------------------------------------

with st.form("rag_query_form"):

    question = st.text_input(
        "Your question"
    )

    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button(
        "Ask"
    )

    if submitted and question.strip():

        with st.spinner(
            "Sending event and generating answer..."
        ):

            # Send event to Inngest
            event_id = asyncio.run(
                send_rag_query_event(
                    question.strip(),
                    int(top_k),
                )
            )

            # Wait for function to complete
            output = wait_for_run_output(
                event_id
                )
            st.write("DEBUG OUTPUT:")
            st.json(output)
            answer = output.get(
                "answer",
                "",
            )

            sources = output.get(
                "sources",
                [],
            )

        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(
            answer or "(No answer)"
        )

        # -------------------------------------------------
        # Display sources
        # -------------------------------------------------

        if sources:

            st.caption("Sources")

            for source in sources:

                st.write(
                    f"- {source}"
                )