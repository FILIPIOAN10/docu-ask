"""Command line: load a folder of documents, or ask one question, without starting the web server.

python -m docuask.cli ingest sample-docs
python -m docuask.cli ask "How many vacation days do I get?"
"""

import argparse
import json
import sys
from pathlib import Path

from docuask.config import get_settings
from docuask.db import get_session_factory
from docuask.ingest import ingest
from docuask.providers import get_embeddings


def cmd_ingest(folder: str) -> int:
    settings, sessions, embeddings = get_settings(), get_session_factory(), get_embeddings()
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".md", ".txt", ".pdf"))
    for path in files:
        result = ingest(sessions, embeddings, settings, path.name, path.read_bytes())
        print(f"{result.status:<10} {path.name} ({result.document.chunk_count} chunks)")
    print(f"{len(files)} files")
    return 0


def cmd_ask(question: str) -> int:
    from docuask.api import get_rag_service  # imported here so `ingest` works before the web layer exists (Phase G)

    answer = get_rag_service().ask(question)
    print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="docuask")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest").add_argument("folder")
    sub.add_parser("ask").add_argument("question")
    args = parser.parse_args()
    return cmd_ingest(args.folder) if args.command == "ingest" else cmd_ask(args.question)


if __name__ == "__main__":
    sys.exit(main())
