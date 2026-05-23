"""CLI entrypoint cho RAG Q&A.

Usage:
    python -m apps.qa <youtube_url> <question>
    python -m apps.qa <url> <question> --provider google --web auto
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tubenote.config import CFG
from tubenote.llm.providers import make_llm
from tubenote.ollama_runtime import ensure_running
from tubenote.pipeline.qa import QAPipeline
from tubenote.processors.rag.pipeline import ingest


def main() -> None:
    p = argparse.ArgumentParser(description="Q&A trên video YouTube qua RAG.")
    p.add_argument("url", help="Link YouTube")
    p.add_argument("question", nargs="+", help="Câu hỏi")
    p.add_argument("--provider", help="google | openai | anthropic | local")
    p.add_argument("--model", help="Override model name")
    p.add_argument("--api-key", dest="api_key", help="Override API key")
    p.add_argument("--web", choices=["off", "auto", "always"], default="off",
                   help="Web search mode (default off)")
    args = p.parse_args()

    if not ensure_running(CFG.embedding.base_url, wait_seconds=15):
        print(
            "❌ Ollama không start được. Cài tại https://ollama.com/download "
            "hoặc chạy `ollama serve`.",
            file=sys.stderr,
        )
        sys.exit(1)

    question = " ".join(args.question)
    llm = make_llm(provider=args.provider, model=args.model, api_key=args.api_key)

    info = ingest(args.url, on_progress=lambda m: print(m, file=sys.stderr))
    if info["indexed"]:
        print(f"📥 Vừa index {info['video_id']} ({info['chunks']} chunks).", file=sys.stderr)
    else:
        print(f"✅ Dùng index có sẵn cho {info['video_id']}.", file=sys.stderr)

    pipe = QAPipeline(llm)
    print(f"❓ {question}\n", file=sys.stderr)
    result = pipe.run(
        video_id=info["video_id"],
        question=question,
        url=args.url,
        web_mode=args.web,
        on_progress=lambda m: print(m, file=sys.stderr),
    )

    print(f"📚 Top {len(result.chunks_used)} chunks retrieved.\n", file=sys.stderr)
    print(result.answer)


if __name__ == "__main__":
    main()
