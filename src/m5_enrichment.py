from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import json
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def clean_llm_response(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau cực kỳ ngắn gọn bằng tiếng Việt (dưới 10 từ)."},
                    {"role": "user", "content": text},
                ],
                max_tokens=60,
            )
            cleaned = clean_llm_response(resp.choices[0].message.content)
            return cleaned
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")
            
    # Extractive fallback:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt mà đoạn văn có thể trả lời. Hãy nhớ kết thúc mỗi câu hỏi bằng dấu chấm hỏi '?'. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            cleaned = clean_llm_response(resp.choices[0].message.content)
            questions = cleaned.split("\n")
            processed_qs = []
            for q in questions:
                q_strip = q.strip().lstrip("0123456789.-) ")
                if q_strip:
                    if not q_strip.endswith("?"):
                        q_strip += "?"
                    processed_qs.append(q_strip)
            return processed_qs[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")
            
    # Extractive fallback:
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu bằng tiếng Việt."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = clean_llm_response(resp.choices[0].message.content)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")
            
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if OPENAI_API_KEY:
        try:
            import json as _json
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}. Không kèm markdown code blocks, chỉ trả về JSON raw.'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            cleaned = clean_llm_response(resp.choices[0].message.content)
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return _json.loads(cleaned.strip())
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
            
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if OPENAI_API_KEY:
        try:
            import json as _json
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON raw không bọc markdown block:
{
  "summary": "tóm tắt 2-3 câu bằng tiếng Việt",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu bằng tiếng Việt",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=1000,
            )
            cleaned = clean_llm_response(resp.choices[0].message.content)
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return _json.loads(cleaned.strip())
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
            
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã song song hóa bằng ThreadPoolExecutor)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_chunk(index: int, chunk: dict) -> tuple[int, EnrichedChunk]:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return index, EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    enriched_results = [None] * len(chunks)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_chunk, idx, chunk): idx for idx, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            idx, res = future.result()
            enriched_results[idx] = res
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(chunks):
                print(f"  Enriched {completed_count}/{len(chunks)} chunks...", flush=True)

    return enriched_results


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
