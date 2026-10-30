# Aventro Motors RAG - Phase 1

This repository contains Phase 1 of the **Aventro Motors Knowledge Base RAG Project**.
Phase 1 focuses on dataset inspection, deduplication, text extraction, cleaning, chunking, local embeddings, and ChromaDB vector indexing. Phase 2 will implement LangGraph orchestration.

---

## Prerequisites
- **OS**: Windows 10 / 11
- **Python**: Python 3.11 - 3.14 (Installed at `C:\Users\devi.c\AppData\Local\Programs\Python\Python314\python.exe`)
- **Editor**: Visual Studio Code with PowerShell terminal

---

## Quick Setup Instructions (VS Code Terminal)

Open a PowerShell terminal in the project root (`c:\Users\devi.c\RAG Phase1`) and execute the following commands:

### 1. Create a Virtual Environment
```powershell
& "C:\Users\devi.c\AppData\Local\Programs\Python\Python314\python.exe" -m venv .venv
```
*(If `python` is already mapped in your PATH, you can simply run `python -m venv .venv`)*

### 2. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> *Note: If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

### 3. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 4. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Verify the Environment
```powershell
python verify_environment.py
```

---

## Dataset Inspection

To rerun the Phase 1 dataset inspection:
```powershell
python inspect_dataset.py
```
This produces `dataset_inventory_report.md` detailing file counts, SHA-256 hashes, duplicate analysis, and CSV evaluation structure.

---

## Project Structure
```text
RAG Phase1/
├── .gitignore                      # Ignores .venv, chroma_db, caches, and artifacts
├── requirements.txt                # Minimal dependencies for Phase 1
├── README.md                       # Setup and run guide
├── verify_environment.py           # Dependency and import verification script
├── inspect_dataset.py              # Dataset inspection script (Phase 1, Step 1)
├── dataset_inventory_report.md     # Inspection output report
└── Aventro Motors/                 # Source dataset folder
    ├── Aventro Motors.csv          # Evaluation questions and expected answers
    ├── docx/                       # 50 Word documents
    ├── html/                       # 50 HTML documents
    ├── md/                         # 50 Clean Markdown source documents
    ├── pdf/                        # 50 PDF documents
    └── pptx/                       # 50 PowerPoint presentations
```

---

## Dependency Overview

| Package | Purpose |
| :--- | :--- |
| `langchain-text-splitters` | Markdown header & recursive chunking |
| `markdown` | Native markdown parsing & verification |
| `pandas` | CSV inspection, QA pair extraction, evaluation dataframes |
| `sentence-transformers` | Local open-source embedding models (runs locally, no API / Ollama needed) |
| `torch` | Underlying tensor execution engine for local embeddings |
| `chromadb` | Local persistent vector database |
| `pytest` | Unit testing and evaluation test suite |
