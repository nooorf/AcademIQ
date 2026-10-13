"""
src/generation/answer_generator.py

LLM-backed answer generator grounded strictly in retrieved chunks.
Falls back to extractive mode when no API key is set.

  - Explicit error message when both API keys absent.
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from src.utils.preprocessing import expand_query
from src.utils.config_loader import cfg

load_dotenv()

# ── Sentence boundary pattern ─────────────────────────────────────────────────
# Splits on ". " only when followed by a capital letter or end of string,
# avoiding splits on "2.00", "B.S.", "etc." abbreviations.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an academic policy assistant for NUST (National University of Sciences and Technology).

Your job is to answer student questions about university policies, rules, and procedures.

STRICT RULES:
1. Answer ONLY using the handbook excerpts provided below. Do not use any outside knowledge.
2. If the answer is not present in the excerpts, say exactly: "I could not find this information in the provided handbook sections."
3. Be concise and direct. Students want clear answers, not walls of text.
4. Always end your answer with a Sources line citing the section name and page number of every excerpt you used.
5. If multiple excerpts contradict each other, mention both and note the discrepancy.

Format your response as:
[Your answer here]

Sources: [Section Name, Page X] | [Section Name, Page X] | ...
"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i} — Section: {chunk.get('section', 'Unknown')}, "
            f"Page {chunk.get('page', '?')}]\n{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


# ── Generator class ───────────────────────────────────────────────────────────

class AnswerGenerator:

    def __init__(self) -> None:
        openai_key    = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        self.model      = os.getenv("LLM_MODEL", cfg("generation", "model", "gpt-4o-mini"))
        self.max_tokens = cfg("generation", "max_tokens",  800)
        self.temperature = cfg("generation", "temperature", 0.1)
        self.client     = None
        self.mode       = "extractive"
        self._backend   = None

        if openai_key:
            try:
                from openai import OpenAI
                self.client  = OpenAI(api_key=openai_key)
                self.mode    = "llm"
                self._backend = "openai"
                print(f"AnswerGenerator: LLM mode via OpenAI ({self.model})")
            except ImportError:
                print("AnswerGenerator: openai package not installed — "
                      "falling back to extractive mode.")
            except Exception as e:
                # Mask key from traceback
                print(f"AnswerGenerator: OpenAI init failed ({type(e).__name__}) — "
                      "falling back to extractive mode.")

        elif anthropic_key:
            try:
                import anthropic
                self.client   = anthropic.Anthropic(api_key=anthropic_key)
                self.mode     = "llm"
                self._backend = "anthropic"
                # Default to a capable Claude model if LLM_MODEL not set
                if self.model == "gpt-4o-mini":
                    self.model = "claude-3-5-haiku-20241022"
                print(f"AnswerGenerator: LLM mode via Anthropic ({self.model})")
            except ImportError:
                print("AnswerGenerator: anthropic package not installed — "
                      "falling back to extractive mode.")
            except Exception as e:
                print(f"AnswerGenerator: Anthropic init failed ({type(e).__name__}) — "
                      "falling back to extractive mode.")

        else:
            print("AnswerGenerator: no API key found (OPENAI_API_KEY / ANTHROPIC_API_KEY) — "
                  "running in extractive mode.  Add key to .env for LLM answers.")

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate an answer grounded in the provided chunks.

        Returns:
            dict with keys: answer, mode, model, chunks_used, latency_ms
        """
        if not chunks:
            return {
                "answer":      "No relevant handbook sections were retrieved for this query.",
                "mode":        self.mode,
                "model":       self.model if self.mode == "llm" else "extractive",
                "chunks_used": 0,
                "latency_ms":  0.0,
            }

        t0 = time.perf_counter()

        if self.mode == "llm":
            answer = self._generate_llm(query, chunks)
        else:
            answer = self._generate_extractive(query, chunks)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer":      answer,
            "mode":        self.mode,
            "model":       self.model if self.mode == "llm" else "extractive",
            "chunks_used": len(chunks),
            "latency_ms":  round(latency_ms, 2),
        }

    # ── LLM mode ───────────────────────────────────────────────────────────

    def _generate_llm(self, query: str, chunks: list[dict]) -> str:
        context      = _build_context(chunks)
        user_message = (
            f"Handbook excerpts:\n\n{context}\n\n"
            f"Student question: {query}"
        )
        try:
            if self._backend == "openai":
                response = self.client.chat.completions.create(
                    model      = self.model,
                    messages   = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    max_tokens  = self.max_tokens,
                    temperature = self.temperature,
                )
                return response.choices[0].message.content.strip()

            elif self._backend == "anthropic":
                response = self.client.messages.create(
                    model      = self.model,
                    system     = SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": user_message}],
                    max_tokens = self.max_tokens,
                )
                return response.content[0].text.strip()

        except Exception as e:
            print(f"LLM API error ({type(e).__name__}) — falling back to extractive.")
            return self._generate_extractive(query, chunks)

        return self._generate_extractive(query, chunks)

    # ── Extractive mode ────────────────────────────────────────────────────

    def _generate_extractive(self, query: str, chunks: list[dict]) -> str:
        """
        Extractive fallback: scan ALL chunks for the most query-relevant
        sentences and return the top 5.

        FIX: was reading only chunks[0]. Now scans all provided chunks and
        also uses expanded query keywords so "gpa" matches "cgpa", "2.00".

        Improvements over prior version:
          - Density scoring (hits / sentence_word_count) to avoid bias
            towards long noisy sentences
          - Sentence deduplication to handle overlapping chunks
          - Returns 5 sentences instead of 3 for richer context
          - More robust sentence splitting
        """
        # Build keyword set: original words + expanded synonyms (len >= 2)
        expanded    = expand_query(query)
        query_words = {
            w.lower() for w in expanded.split()
            if len(w) >= 2
        }

        # Score every sentence across ALL chunks
        scored_sentences: list[tuple[float, str, str]] = []  # (score, sent, section)
        seen_sentences: set[str] = set()  # deduplication

        for chunk in chunks:
            text = chunk.get("text") or ""
            section = chunk.get("section", "Unknown")
            if not text:
                continue

            # More robust sentence splitting:
            # Split on period/question/excl followed by whitespace,
            # but also split on numbered list items like (1), (2), (a), (b)
            raw_text = text.replace('\n', ' ')
            # First split on sentence boundaries
            raw_sentences = _SENT_SPLIT_RE.split(raw_text)
            # Then further split long segments on numbered items
            sentences = []
            for seg in raw_sentences:
                # Split on patterns like "(1) ", "(a) ", "(i) "
                parts = re.split(r'(?=\(\d+\)\s|\([a-z]\)\s)', seg)
                sentences.extend(parts)

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 25:
                    continue
                # Deduplication: skip near-duplicate sentences
                sent_key = ' '.join(sent.lower().split()[:12])
                if sent_key in seen_sentences:
                    continue
                seen_sentences.add(sent_key)

                sent_words = sent.lower().split()
                hits = sum(1 for w in query_words if w in sent.lower())
                if hits > 0:
                    # Density scoring: hits normalized by sentence length
                    # avoids bias towards long noisy sentences
                    density = hits / max(len(sent_words), 1)
                    # Combine raw hits + density for balanced scoring
                    score = hits + (density * 3)
                    scored_sentences.append((score, sent, section))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences:
            # Take top 5 sentences for richer extractive context
            top_sentences = scored_sentences[:5]
            answer_text = " ".join(s for _, s, _ in top_sentences)
            if not answer_text.endswith('.'):
                answer_text += "."
        else:
            # Hard fallback: first 80 words of the top-ranked chunk
            fallback_text = (chunks[0].get("text") or "")
            answer_text   = " ".join(fallback_text.split()[:80]) + "..."

        sources = " | ".join(
            f"{c.get('section', 'Unknown')}, Page {c.get('page', '?')}"
            for c in chunks
        )
        return f"{answer_text}\n\nSources: {sources}"