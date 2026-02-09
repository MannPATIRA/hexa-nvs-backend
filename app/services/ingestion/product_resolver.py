"""
Multi-stage product resolver for matching supplier descriptions to canonical products.

Matching cascade (cheapest → most expensive):
  1. Exact supplier code lookup
  2. Exact alias text lookup
  3. Semantic search (embedding similarity via pgvector + keyword scoring)
  4. LLM judge verification (optional, for ambiguous matches)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI
from rapidfuzz import fuzz
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class MatchCandidate:
    """An intermediate candidate found by semantic/keyword search."""
    product_id: str
    product_name: str
    matched_text: str
    category: str | None = None
    unit: str | None = None
    pack_size: int | None = None
    embedding_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class MatchResult:
    """Final result returned by the resolver."""
    product_id: str | None
    product_name: str | None
    confidence: float
    strategy: str  # "exact_code", "exact_alias", "semantic", "llm_verified", "none"
    reasoning: str | None = None


# ═══════════════════════════════════════════════════════════════
# Portuguese stop words and unit synonyms
# ═══════════════════════════════════════════════════════════════

_PT_STOP_WORDS: set[str] = {
    "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos", "nas",
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "com", "por", "para",
    "ao", "aos", "se", "que", "ou", "mais", "tipo", "c",
}

_UNIT_SYNONYMS: dict[str, str] = {
    "kg": "kg", "kilos": "kg", "kilo": "kg", "quilos": "kg", "quilo": "kg",
    "g": "g", "gramas": "g", "grama": "g", "gr": "g",
    "l": "l", "litros": "l", "litro": "l", "lt": "l",
    "ml": "ml", "mililitros": "ml",
    "cx": "cx", "caixa": "cx", "caixas": "cx",
    "un": "un", "unid": "un", "unidade": "un", "unidades": "un",
    "sc": "sc", "saco": "sc", "sacos": "sc",
    "pct": "pct", "pacote": "pct", "pacotes": "pct",
    "fd": "fd", "fardo": "fd", "fardos": "fd",
    "dz": "dz", "duzia": "dz", "duzias": "dz",
    "pc": "pc", "peca": "pc", "pecas": "pc", "peça": "pc", "peças": "pc",
}


# ═══════════════════════════════════════════════════════════════
# Resolver
# ═══════════════════════════════════════════════════════════════

class ProductResolver:
    """
    Resolve a raw supplier description to a canonical product using a
    multi-stage cascade: exact lookups → embedding + keyword search → LLM judge.
    """

    # ── Models ──────────────────────────────────────────────
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 512
    LLM_MODEL = "gpt-4o-mini"

    # ── Thresholds ──────────────────────────────────────────
    HIGH_CONFIDENCE = 0.90   # Skip LLM — return immediately
    MATCH_THRESHOLD = 0.50   # Minimum combined score to consider
    TOP_K = 5                # Candidates sent to LLM judge

    # ── Score weights ───────────────────────────────────────
    EMBEDDING_WEIGHT = 0.65
    KEYWORD_WEIGHT = 0.35

    def __init__(
        self,
        session: AsyncSession,
        openai_client: AsyncOpenAI,
        *,
        use_llm_verification: bool = True,
    ):
        self.session = session
        self.openai = openai_client
        self.use_llm_verification = use_llm_verification

    # ════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════

    async def resolve(
        self,
        raw_description: str,
        supplier_id: str,
        supplier_code: str | None = None,
    ) -> MatchResult:
        """
        Match *raw_description* to a canonical product.

        Stages tried in order:
          1. Exact supplier-code lookup   → confidence 1.0
          2. Exact alias text lookup      → confidence 1.0
          3. Semantic + keyword search    → combined score
          4. LLM judge (if enabled)       → refined confidence
        """

        # Stage 1 — exact code
        if supplier_code:
            result = await self._match_by_code(supplier_code, supplier_id)
            if result:
                return result

        # Stage 2 — exact alias
        result = await self._match_by_exact_alias(raw_description, supplier_id)
        if result:
            return result

        # Stage 3 — semantic + keyword
        candidates = await self._semantic_search(raw_description)

        if not candidates:
            return MatchResult(
                product_id=None, product_name=None,
                confidence=0.0, strategy="none",
            )

        best = candidates[0]

        # High-confidence hit — return directly
        if best.combined_score >= self.HIGH_CONFIDENCE:
            return MatchResult(
                product_id=best.product_id,
                product_name=best.product_name,
                confidence=round(best.combined_score, 3),
                strategy="semantic",
            )

        # Medium-confidence — try LLM verification
        if (
            self.use_llm_verification
            and best.combined_score >= self.MATCH_THRESHOLD
        ):
            llm_result = await self._llm_verify(raw_description, candidates[: self.TOP_K])
            if llm_result:
                return llm_result

        # Return best candidate if it clears the threshold (no LLM or LLM disabled)
        if best.combined_score >= self.MATCH_THRESHOLD and not self.use_llm_verification:
            return MatchResult(
                product_id=best.product_id,
                product_name=best.product_name,
                confidence=round(best.combined_score, 3),
                strategy="semantic",
            )

        # No match
        return MatchResult(
            product_id=None, product_name=None,
            confidence=0.0, strategy="none",
        )

    async def ensure_embeddings(self) -> int:
        """
        Generate and store embeddings for every product and alias that
        doesn't have one yet.  Returns the count of newly created embeddings.
        """
        count = 0

        # ── Products without embeddings ─────────────────────
        rows = await self.session.execute(text("""
            SELECT id, canonical_name
            FROM products
            WHERE embedding IS NULL
        """))
        products = rows.mappings().all()
        if products:
            texts = [r["canonical_name"] for r in products]
            embeddings = await self._get_embeddings_batch(texts)
            for row, emb in zip(products, embeddings):
                vec_literal = _vec_literal(emb)
                await self.session.execute(
                    text("UPDATE products SET embedding = :emb WHERE id = :id"),
                    {"emb": vec_literal, "id": row["id"]},
                )
                count += 1

        # ── Aliases without embeddings ──────────────────────
        rows = await self.session.execute(text("""
            SELECT id, raw_description
            FROM product_aliases
            WHERE embedding IS NULL
        """))
        aliases = rows.mappings().all()
        if aliases:
            texts = [r["raw_description"] for r in aliases]
            embeddings = await self._get_embeddings_batch(texts)
            for row, emb in zip(aliases, embeddings):
                vec_literal = _vec_literal(emb)
                await self.session.execute(
                    text("UPDATE product_aliases SET embedding = :emb WHERE id = :id"),
                    {"emb": vec_literal, "id": row["id"]},
                )
                count += 1

        await self.session.commit()
        logger.info("ensure_embeddings: generated %d new embeddings", count)
        return count

    # ════════════════════════════════════════════════════════
    # Stage 1 + 2 — exact lookups (unchanged logic)
    # ════════════════════════════════════════════════════════

    async def _match_by_code(
        self, supplier_code: str, supplier_id: str,
    ) -> MatchResult | None:
        result = await self.session.execute(text("""
            SELECT pa.product_id, p.canonical_name
            FROM product_aliases pa
            JOIN products p ON p.id = pa.product_id
            WHERE pa.supplier_code = :code AND pa.supplier_id = :sid
            LIMIT 1
        """), {"code": supplier_code, "sid": supplier_id})
        row = result.mappings().first()
        if row:
            return MatchResult(
                product_id=str(row["product_id"]),
                product_name=row["canonical_name"],
                confidence=1.0,
                strategy="exact_code",
            )
        return None

    async def _match_by_exact_alias(
        self, raw_description: str, supplier_id: str,
    ) -> MatchResult | None:
        result = await self.session.execute(text("""
            SELECT pa.product_id, p.canonical_name
            FROM product_aliases pa
            JOIN products p ON p.id = pa.product_id
            WHERE pa.raw_description = :raw AND pa.supplier_id = :sid
            LIMIT 1
        """), {"raw": raw_description, "sid": supplier_id})
        row = result.mappings().first()
        if row:
            return MatchResult(
                product_id=str(row["product_id"]),
                product_name=row["canonical_name"],
                confidence=1.0,
                strategy="exact_alias",
            )
        return None

    # ════════════════════════════════════════════════════════
    # Stage 3 — semantic + keyword search
    # ════════════════════════════════════════════════════════

    async def _semantic_search(self, raw_description: str) -> list[MatchCandidate]:
        """
        1. Embed the query.
        2. pgvector cosine-similarity search for top-K candidates
           (searches both product canonical names and aliases).
        3. Re-score each candidate with a keyword overlap metric.
        4. Combine embedding + keyword scores.
        """
        query_embedding = await self._get_embedding(raw_description)
        if not query_embedding:
            return []

        vec_literal = _vec_literal(query_embedding)

        # Cosine similarity search across aliases and products.
        # The <=> operator returns *distance* (0 = identical), so
        # similarity = 1 - distance.
        result = await self.session.execute(text("""
            (
                SELECT
                    pa.product_id,
                    p.canonical_name AS product_name,
                    pa.raw_description AS matched_text,
                    p.category,
                    p.unit,
                    p.pack_size,
                    1 - (pa.embedding <=> CAST(:qvec AS vector)) AS similarity
                FROM product_aliases pa
                JOIN products p ON p.id = pa.product_id
                WHERE pa.embedding IS NOT NULL
            )
            UNION ALL
            (
                SELECT
                    p.id AS product_id,
                    p.canonical_name AS product_name,
                    p.canonical_name AS matched_text,
                    p.category,
                    p.unit,
                    p.pack_size,
                    1 - (p.embedding <=> CAST(:qvec AS vector)) AS similarity
                FROM products p
                WHERE p.embedding IS NOT NULL
            )
            ORDER BY similarity DESC
            LIMIT :topk
        """), {"qvec": vec_literal, "topk": self.TOP_K * 2})

        rows = result.mappings().all()
        if not rows:
            return []

        # De-duplicate by product_id — keep the row with highest similarity
        seen: dict[str, MatchCandidate] = {}
        for row in rows:
            pid = str(row["product_id"])
            emb_score = float(row["similarity"])
            if pid in seen and seen[pid].embedding_score >= emb_score:
                continue
            seen[pid] = MatchCandidate(
                product_id=pid,
                product_name=row["product_name"],
                matched_text=row["matched_text"],
                category=row["category"],
                unit=row["unit"],
                pack_size=row["pack_size"],
                embedding_score=emb_score,
            )

        # Keyword scoring + combined score
        candidates = list(seen.values())
        for c in candidates:
            c.keyword_score = self._keyword_score(raw_description, c.matched_text)
            c.combined_score = (
                self.EMBEDDING_WEIGHT * c.embedding_score
                + self.KEYWORD_WEIGHT * c.keyword_score
            )

        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[: self.TOP_K]

    # ════════════════════════════════════════════════════════
    # Stage 4 — LLM judge
    # ════════════════════════════════════════════════════════

    async def _llm_verify(
        self,
        raw_description: str,
        candidates: list[MatchCandidate],
    ) -> MatchResult | None:
        """
        Ask an LLM to pick the best match (or reject all) from a
        shortlist of candidates.
        """
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            parts = [f"{i}. \"{c.product_name}\""]
            if c.category:
                parts.append(f"category={c.category}")
            if c.unit:
                parts.append(f"unit={c.unit}")
            if c.pack_size:
                parts.append(f"pack_size={c.pack_size}")
            parts.append(f"(score={c.combined_score:.2f})")
            candidate_lines.append("  ".join(parts))

        candidates_text = "\n".join(candidate_lines)

        system_prompt = (
            "You are a product-matching expert for a Brazilian food distributor.\n"
            "You will be given a raw product description from a supplier price list "
            "and a numbered list of candidate products from the internal catalogue.\n\n"
            "Your task:\n"
            "- Decide which candidate (if any) is the SAME product as the raw description.\n"
            "- Products may differ in wording, abbreviations, order, or language "
            "but must refer to the same physical item (same type, same pack size, same unit).\n"
            "- If none of the candidates match, say so.\n\n"
            "Respond with ONLY a JSON object (no markdown fences):\n"
            '{"match": <number 1-N or null>, "confidence": <0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )

        user_prompt = (
            f"Raw description from supplier:\n\"{raw_description}\"\n\n"
            f"Candidate products:\n{candidates_text}"
        )

        try:
            response = await self.openai.chat.completions.create(
                model=self.LLM_MODEL,
                temperature=0.0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            # Strip markdown fences if the model adds them anyway
            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            verdict = json.loads(content)
        except Exception:
            logger.warning("LLM verification failed for '%s'", raw_description, exc_info=True)
            return None

        match_idx = verdict.get("match")
        confidence = float(verdict.get("confidence", 0))
        reasoning = verdict.get("reasoning", "")

        if match_idx is None or confidence < self.MATCH_THRESHOLD:
            return None

        # match_idx is 1-based
        if not (1 <= match_idx <= len(candidates)):
            return None

        chosen = candidates[match_idx - 1]
        return MatchResult(
            product_id=chosen.product_id,
            product_name=chosen.product_name,
            confidence=round(confidence, 3),
            strategy="llm_verified",
            reasoning=reasoning,
        )

    # ════════════════════════════════════════════════════════
    # Embedding helpers
    # ════════════════════════════════════════════════════════

    async def _get_embedding(self, text_input: str) -> list[float] | None:
        """Get a single embedding vector from OpenAI."""
        try:
            resp = await self.openai.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text_input,
                dimensions=self.EMBEDDING_DIMENSIONS,
            )
            return resp.data[0].embedding
        except Exception:
            logger.warning("Embedding request failed for: %s", text_input[:80], exc_info=True)
            return None

    async def _get_embeddings_batch(
        self, texts: list[str], batch_size: int = 100,
    ) -> list[list[float]]:
        """Get embeddings for a list of texts, batching API calls."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await self.openai.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=batch,
                dimensions=self.EMBEDDING_DIMENSIONS,
            )
            # API returns results in order
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    # ════════════════════════════════════════════════════════
    # Keyword scoring
    # ════════════════════════════════════════════════════════

    @classmethod
    def _keyword_score(cls, query: str, candidate: str) -> float:
        """
        Combined keyword score:
          50 % token Jaccard similarity (normalised, stop-words removed)
          50 % RapidFuzz token_sort_ratio (handles reordering + typos)
        """
        q_tokens = cls._tokenize(query)
        c_tokens = cls._tokenize(candidate)

        # Jaccard
        if q_tokens or c_tokens:
            intersection = q_tokens & c_tokens
            union = q_tokens | c_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
        else:
            jaccard = 0.0

        # RapidFuzz (operates on normalised strings, not token sets)
        q_norm = cls._normalise(query)
        c_norm = cls._normalise(candidate)
        fuzzy = fuzz.token_sort_ratio(q_norm, c_norm) / 100.0

        return 0.5 * jaccard + 0.5 * fuzzy

    # ════════════════════════════════════════════════════════
    # Text normalisation & tokenisation
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _normalise(text_input: str) -> str:
        """Normalise text for comparison: lowercase, strip punctuation, standardise units."""
        t = text_input.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t)      # Remove punctuation
        t = re.sub(r"\s+", " ", t)           # Collapse whitespace

        # Standardise unit synonyms in-place
        tokens = t.split()
        normalised = []
        for tok in tokens:
            normalised.append(_UNIT_SYNONYMS.get(tok, tok))
        return " ".join(normalised).strip()

    @staticmethod
    def _tokenize(text_input: str) -> set[str]:
        """
        Tokenize text into a set of meaningful tokens:
        lowercased, punctuation removed, stop words removed,
        units normalised.
        """
        t = text_input.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        tokens = set()
        for tok in t.split():
            tok = _UNIT_SYNONYMS.get(tok, tok)
            if tok not in _PT_STOP_WORDS and len(tok) > 1:
                tokens.add(tok)
        return tokens


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _vec_literal(embedding: list[float]) -> str:
    """
    Format a Python list of floats into the pgvector text literal
    that PostgreSQL expects, e.g. '[0.1,0.2,0.3]'.
    """
    inner = ",".join(f"{v:.8f}" for v in embedding)
    return f"[{inner}]"
