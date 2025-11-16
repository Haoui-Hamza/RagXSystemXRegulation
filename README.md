# RagXSystemXRegulation

A Python toolkit to compare an operator's Maintenance Organisation Exposition (MOE) against EASA Part‑145 using embeddings and a chat LLM.

The repository ingests a Part‑145 PDF and an MOE DOCX, builds searchable embeddings with Chroma, retrieves relevant regulatory excerpts for each MOE section, and asks a chat model to classify compliance and suggest improvements. Outputs are written under `results/` as JSON summaries and per-section findings.

the repo for the frontend is here : https://github.com/anesszereg/junction_2025/tree/main

## Key files

- `code.py` — main script containing ingestion, index-building, retrieval and LLM prompt logic.
- `part145_db/` and `chroma_db/` — Chroma database folders used for persistent embeddings.
- `results/` — output directory. Contains `master_summary.json` (aggregate stats) and `sections/` with per-section JSON produced by the LLM.

## Prerequisites

- Python 3.10+ (tested locally on macOS).
- Environment variable `OPENAI_API_KEY` set to an API key that the OpenAIEmbeddingFunction/OpenAI client can use.
- Libraries (install via pip):

```bash
pip install chromadb pypdf python-docx openai
```

Notes:
- The code imports `OpenAIEmbeddingFunction` from `chromadb.utils.embedding_functions` and `OpenAI` from the `openai` package (modern OpenAI SDK). Ensure the installed package versions are compatible with those imports.

## Quick start

1. Place your EASA Part‑145 PDF next to the repo and update `PART145_PDF_PATH` in `code.py` (default: `part-145.pdf`).
2. Place your MOE DOCX and update `MOE_DOCX_PATH` (default: `AI anonyymi MOE.docx`).
3. Ensure `OPENAI_API_KEY` is set in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

4. Run the script (simple run; `code.py` exposes functions you can call or import):

```bash
python code.py
```

The script contains functions to build indexes and call the LLM. Depending on how you run it, outputs (per-section JSON and `master_summary.json`) will appear under `results/`.

## How it works (brief)

- **build_part145_index(pdf_path)**: Extracts text per page from a Part‑145 PDF, chunks it, and stores embeddings/metadata in the `easa_part145` Chroma collection.
- **build_moe_index(docx_path)**: Loads the MOE DOCX, chunks it into MOE sections, and stores them in the `moe_manual` Chroma collection.
- **retrieve_part145_chunks(query_text, top_k)**: Retrieves the most relevant Part‑145 excerpts for a given MOE chunk.
- **build_moe_comparison_prompt(...)**: Constructs a system + user message pair that asks an LLM to compare a MOE section to regulatory excerpts and return a strict JSON result describing compliance and recommended fixes.

## Outputs

- `results/master_summary.json` — generated-at timestamp, per-section compliance classifications and summary stats.
- `results/sections/*.json` — individual LLM JSON outputs for MOE sections (IDs like `MOE-0001`, ...).

Example of `master_summary.json` structure (already present in the repo):

```json
{
  "generated_at": "2025-11-15T15:58:15.605766",
  "num_sections": 13,
  "sections": [ ... ],
  "stats": { "compliant": 0, "partially_compliant": 12, "non_compliant": 1, "missing": 0 }
}
```




