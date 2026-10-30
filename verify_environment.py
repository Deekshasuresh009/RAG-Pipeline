"""
Environment and Dependency Verification Script
Aventro Motors RAG Project - Phase 1, Step 2
Verifies that all required packages for Markdown processing, CSV handling,
local embeddings, ChromaDB, and testing can be imported cleanly.
"""

import sys

REQUIRED_PACKAGES = [
    ("langchain_text_splitters", "langchain-text-splitters", "Markdown chunking & splitting"),
    ("markdown", "markdown", "Markdown parsing"),
    ("pandas", "pandas", "CSV inspection & data analysis"),
    ("torch", "torch", "Local neural network / tensor computation"),
    ("sentence_transformers", "sentence-transformers", "Local HuggingFace embeddings"),
    ("chromadb", "chromadb", "Local vector database"),
    ("pytest", "pytest", "Test framework"),
]


def test_imports():
    print("=" * 70)
    print("   AVENTRO MOTORS RAG - PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print("-" * 70)

    success_count = 0
    missing_packages = []

    for import_name, pkg_name, purpose in REQUIRED_PACKAGES:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f"[OK]   {pkg_name:<25} (v{version}) - {purpose}")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {pkg_name:<25} - NOT INSTALLED ({purpose})")
            missing_packages.append((pkg_name, str(e)))

    print("-" * 70)
    if missing_packages:
        print(f"Result: {success_count}/{len(REQUIRED_PACKAGES)} packages verified.")
        print(f"Pending installations ({len(missing_packages)}):")
        for pkg, err in missing_packages:
            print(f"  - {pkg}")
        print("\nTo install, activate your virtual environment (.venv) and run:")
        print("    pip install -r requirements.txt")
        return False
    else:
        print(f"Result: All {success_count}/{len(REQUIRED_PACKAGES)} packages imported successfully!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
