import os
import uuid
import json
import datetime
import re
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pypdf import PdfReader
import docx
from openai import OpenAI

# =========================
# Configuration
# =========================

PART145_PDF_PATH = "part-145.pdf"          # <-- put your actual Part-145 PDF here
MOE_DOCX_PATH = "AI anonyymi MOE.docx"      # <-- or rename and update this path

CHROMA_PATH = "chroma_db"
PART145_COLLECTION_NAME = "easa_part145"
MOE_COLLECTION_NAME = "moe_manual"

EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.1"  # or another chat model you have access to

RESULTS_DIR = "results"
SECTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "sections")

client = OpenAI()


# =========================
# Generic helpers
# =========================

def split_into_chunks(text: str, max_tokens: int = 1500) -> List[str]:
    """
    Robust chunker: splits by words, independent of paragraphs.
    Guarantees chunks are always below max_tokens (~max model context).
    """
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks



def get_or_create_collection(name: str):
    """
    Initialize Chroma client and get/create the collection with OpenAI embeddings.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )

    collection = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
    )

    return collection


# =========================
# Part-145 PDF ingestion
# =========================

def load_pdf_text(pdf_path: str) -> List[str]:
    """
    Extract text per page from a PDF.
    Returns a list where each element is the text of one page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Part-145 PDF not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        pages_text.append(text or "")

    return pages_text


def build_part145_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from the PDF and creates a list of chunk dicts
    with metadata: id, text, page, and a naive section guess.
    """
    pages_text = load_pdf_text(pdf_path)
    chunks_with_meta = []

    for page_idx, page_text in enumerate(pages_text):
        page_num = page_idx + 1

        # Very naive "section" detection: look for lines starting with "145."
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        possible_section = None
        for line in lines:
            if line.startswith("145.") or "Part-145" in line or "PART-145" in line:
                possible_section = line
                break

        chunks = split_into_chunks(page_text, max_tokens=1500)

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunks_with_meta.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "page": page_num,
                    "section": possible_section or "Unknown section",
                }
            )

    return chunks_with_meta


def build_part145_index(pdf_path: str) -> None:
    """
    Build the Chroma index for EASA Part-145 from the PDF.
    """
    print(f"[Part145] Loading and chunking PDF: {pdf_path}")
    chunks = build_part145_chunks(pdf_path)
    print(f"[Part145] Prepared {len(chunks)} chunks.")

    collection = get_or_create_collection(PART145_COLLECTION_NAME)

    # Optionally clear collection if re-building:
    # existing = collection.get()
    # if existing["ids"]:
    #     collection.delete(ids=existing["ids"])

    print("[Part145] Inserting chunks into Chroma...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [{"page": c["page"], "section": c["section"]} for c in batch]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print("[Part145] Index built successfully.")


def retrieve_part145_chunks(
    query_text: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant Part-145 chunks for a given text.
    """
    collection = get_or_create_collection(PART145_COLLECTION_NAME)

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved = []
    for doc, meta in zip(docs, metadatas):
        retrieved.append(
            {
                "text": doc,
                "page": meta.get("page"),
                "section": meta.get("section"),
            }
        )

    return retrieved


# =========================
# MOE DOCX ingestion
# =========================

def load_docx_text(docx_path: str) -> str:
    """
    Load all paragraph text from a DOCX file into a single string.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"MOE DOCX not found at: {docx_path}")

    doc = docx.Document(docx_path)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def build_moe_chunks(docx_path: str) -> List[Dict[str, Any]]:
    """
    Build MOE chunks with basic metadata (id, chunk text, order, section_hint).
    """
    full_text = load_docx_text(docx_path)
    chunks = split_into_chunks(full_text, max_tokens=1500)

    chunks_with_meta = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"MOE-{idx+1:04d}"
        first_line = chunk.strip().split("\n")[0][:120]
        chunks_with_meta.append(
            {
                "id": chunk_id,
                "text": chunk,
                "order": idx,
                "section_hint": first_line,
            }
        )

    return chunks_with_meta


def build_moe_index(docx_path: str) -> None:
    """
    Build the Chroma index for the operator's MOE manual (DOCX).
    """
    print(f"[MOE] Loading and chunking DOCX: {docx_path}")
    chunks = build_moe_chunks(docx_path)
    print(f"[MOE] Prepared {len(chunks)} chunks.")

    collection = get_or_create_collection(MOE_COLLECTION_NAME)

    # Optionally clear if re-building:
    # existing = collection.get()
    # if existing["ids"]:
    #     collection.delete(ids=existing["ids"])

    print("[MOE] Inserting chunks into Chroma...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {"order": c["order"], "section_hint": c["section_hint"]}
            for c in batch
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print("[MOE] Index built successfully.")


def get_all_moe_chunks() -> Dict[str, Any]:
    """
    Retrieve all MOE chunks (ids, documents, metadatas) from Chroma.
    """
    collection = get_or_create_collection(MOE_COLLECTION_NAME)
    data = collection.get()
    return data


# =========================
# LLM prompts for MOE vs Part-145
# =========================

def build_moe_comparison_prompt(
    moe_text: str,
    moe_meta: Dict[str, Any],
    part145_chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build the messages (system + user) for the chat model to compare
    one MOE section against relevant Part-145 regulation.
    """

    moe_id = moe_meta.get("id", "unknown-id")
    section_hint = moe_meta.get("section_hint", "")

    system_message = """
You are an aviation compliance auditor specialized in EASA Part-145.

You receive:
1) A section/paragraph from an operator’s Maintenance Organisation Exposition (MOE).
2) A set of relevant excerpts from EASA Part-145.

Your task is to:
- Identify which regulatory requirements in the excerpts apply to the MOE section.
- Determine how well the MOE section satisfies those requirements.
- Classify overall compliance for this MOE section as:
  - "compliant"
  - "partially_compliant"
  - "non_compliant"
  - or "missing" (if the MOE section does not address the requirement at all).
- Explain your reasoning clearly and concisely.
- Suggest concrete text-level improvements to the MOE section when needed.

Output your answer strictly in the following JSON format (NO extra text):

{
  "moe_section_id": "string (e.g. MOE-0001)",
  "moe_section_title": "short human-readable label for this MOE section",
  "summary": "short high level summary of what the MOE section says",
  "overall_compliance": "compliant | partially_compliant | non_compliant | missing",
  "findings": [
    {
      "requirement_id": "free text identifier, e.g. '145.A.30(b) page 23'",
      "requirement_excerpt": "short excerpt from the provided regulatory text",
      "compliance": "compliant | partially_compliant | non_compliant | unclear",
      "reasoning": "why you reached this conclusion",
      "risk": "low | medium | high",
      "recommended_fix": "specific suggestion to improve the MOE text (or empty if compliant)"
    }
  ]
}
    """.strip()

    context_lines = []
    context_lines.append(f"MOE SECTION ID: {moe_id}")
    if section_hint:
        context_lines.append(f"MOE SECTION HINT: {section_hint}")
    context_lines.append("")
    context_lines.append("MOE SECTION TEXT:")
    context_lines.append("---")
    context_lines.append(moe_text)
    context_lines.append("")
    context_lines.append("RELEVANT EASA PART-145 EXCERPTS:")
    context_lines.append("---")

    for idx, chunk in enumerate(part145_chunks, start=1):
        context_lines.append(
            f"{idx}) [section: {chunk.get('section')}, page: {chunk.get('page')}]"
        )
        context_lines.append(chunk["text"])
        context_lines.append("")

    context_lines.append("INSTRUCTIONS:")
    context_lines.append("- Only use the provided regulatory excerpts when interpreting EASA Part-145.")
    context_lines.append("- If something is unclear or not covered by the excerpts, mark the finding as 'unclear' and explain.")
    context_lines.append("- Follow the JSON schema defined in the system message.")
    context_lines.append("- DO NOT include any text outside the JSON object.")

    user_message = "\n".join(context_lines)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call the OpenAI chat model and return the raw content string.
    """
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return completion.choices[0].message.content


def extract_json_from_output(raw_output: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM output.
    Raises if parsing fails.
    """
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM output")

    json_text = json_match.group(0)

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        print("---- RAW JSON TEXT ----")
        print(json_text)
        print("-----------------------")
        raise ValueError(f"Invalid JSON from LLM: {e}")

    return obj


# =========================
# Saving results
# =========================

def save_section_result_json(
    moe_section_id: str,
    json_obj: Dict[str, Any],
    results_dir: str = SECTION_RESULTS_DIR
) -> str:
    """
    Save JSON result for one MOE section into results/sections/<moe_section_id>.json
    """
    os.makedirs(results_dir, exist_ok=True)

    # sanitize filename
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", moe_section_id)
    filename = f"{safe_id}.json"
    file_path = os.path.join(results_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

    return file_path


def save_master_summary_index(results_dir: str = SECTION_RESULTS_DIR) -> str:
    """
    Scan all per-section JSONs and build a simple master index.
    """
    if not os.path.exists(results_dir):
        print("[Summary] No sections folder found, skipping master summary.")
        return ""

    files = [f for f in os.listdir(results_dir) if f.lower().endswith(".json")]
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "num_sections": len(files),
        "sections": [],
        "stats": {
            "compliant": 0,
            "partially_compliant": 0,
            "non_compliant": 0,
            "missing": 0,
            "other": 0,
        }
    }

    for fname in files:
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        sec_id = obj.get("moe_section_id", fname)
        overall = obj.get("overall_compliance", "other")

        summary["sections"].append(
            {
                "moe_section_id": sec_id,
                "overall_compliance": overall,
            }
        )

        if overall in summary["stats"]:
            summary["stats"][overall] += 1
        else:
            summary["stats"]["other"] += 1

    master_path = os.path.join(RESULTS_DIR, "master_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return master_path


# =========================
# Main analysis loop
# =========================

def analyze_moe_vs_part145(
    max_sections: Optional[int] = None,
    top_k: int = 5,
) -> None:
    """
    For each MOE chunk:
      - retrieve relevant Part-145 chunks
      - ask LLM to compare
      - save JSON result per section
    """
    data = get_all_moe_chunks()
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not ids:
        raise RuntimeError(
            "No MOE chunks found in Chroma. Did you run --build-moe-index ?"
        )

    print(f"[Analysis] Found {len(ids)} MOE chunks to analyze.")

    for idx, (doc_id, doc_text, meta) in enumerate(zip(ids, docs, metadatas)):
        if max_sections is not None and idx >= max_sections:
            break

        moe_meta = dict(meta)
        moe_meta["id"] = doc_id

        print(f"[Analysis] ({idx+1}/{len(ids)}) MOE section {doc_id}")

        # Retrieve relevant Part-145 chunks
        part145_chunks = retrieve_part145_chunks(doc_text, top_k=top_k)
        if not part145_chunks:
            print(f"  -> No Part-145 chunks retrieved for {doc_id}, skipping.")
            continue

        # Build prompt and call LLM
        messages = build_moe_comparison_prompt(doc_text, moe_meta, part145_chunks)
        raw_output = call_llm(messages)

        try:
            json_obj = extract_json_from_output(raw_output)
        except Exception as e:
            print(f"  -> Error parsing JSON for {doc_id}: {e}")
            continue

        # Ensure moe_section_id is set
        if "moe_section_id" not in json_obj or not json_obj["moe_section_id"]:
            json_obj["moe_section_id"] = doc_id

        # Save per-section JSON
        saved_path = save_section_result_json(doc_id, json_obj)
        print(f"  -> Saved result to {saved_path}")

    # Build master summary
    master_path = save_master_summary_index()
    if master_path:
        print(f"[Analysis] Master summary saved to {master_path}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOE vs EASA Part-145 RAG Auditor")
    parser.add_argument(
        "--build-part145-index",
        action="store_true",
        help="Build the Chroma index from the EASA Part-145 PDF",
    )
    parser.add_argument(
        "--build-moe-index",
        action="store_true",
        help="Build the Chroma index from the MOE DOCX manual",
    )
    parser.add_argument(
        "--analyze-moe",
        action="store_true",
        help="Run MOE vs Part-145 compliance analysis",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Max number of MOE sections to analyze (for testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of Part-145 chunks to retrieve per MOE section",
    )

    args = parser.parse_args()

    if args.build_part145_index:
        build_part145_index(PART145_PDF_PATH)

    if args.build_moe_index:
        build_moe_index(MOE_DOCX_PATH)

    if args.analyze_moe:
        analyze_moe_vs_part145(
            max_sections=args.max_sections,
            top_k=args.top_k,
        )
