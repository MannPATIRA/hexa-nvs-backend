"""
Ingestion pipeline controller.

Orchestrates:  PDF parsing  →  product resolution  →  DB writes.

Usage from a FastAPI endpoint::

    controller = IngestController(session, tenant_id, openai_client)
    result = await controller.ingest_price_list(raw_bytes, supplier_id)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.parsers.pdf_parser import PdfLlmParser
from app.services.ingestion.product_resolver import ProductResolver
from app.services.ingestion.writer import IngestWriter

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Confidence thresholds
# ═══════════════════════════════════════════════════════════════

CONFIDENCE_AUTO_MATCH = 0.85   # Above this → auto-create price record + alias
CONFIDENCE_SUGGEST = 0.50      # Above this → suggest to buyer (pending match)


# ═══════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class IngestResult:
    document_id: str
    items_extracted: int
    items_matched: int
    items_unresolved: int
    matched_items: list[dict] = field(default_factory=list)
    pending_items: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Controller
# ═══════════════════════════════════════════════════════════════

class IngestController:
    """
    Full ingestion pipeline:

    1. Create a ``documents`` record (status='processing')
    2. Parse the PDF into ``ExtractedLineItem`` objects
    3. For each item, resolve against the canonical product catalog
    4. High-confidence matches  → ``price_records`` + ``product_aliases``
    5. Lower-confidence matches → ``pending_matches`` for buyer review
    6. Update the ``documents`` row with final stats
    """

    def __init__(
        self,
        session: AsyncSession,
        tenant_id: str,
        openai_client: AsyncOpenAI,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self.resolver = ProductResolver(
            session=session,
            openai_client=openai_client,
            use_llm_verification=True,
        )
        self.writer = IngestWriter(session)

    # ── Public API ────────────────────────────────────────────

    async def ingest_price_list(
        self,
        raw_bytes: bytes,
        supplier_id: str,
        supplier_terms: str | None = None,
        max_pages: int | None = None,
    ) -> IngestResult:
        """
        Run the full ingestion pipeline and return structured results.

        Parameters
        ----------
        raw_bytes : bytes
            Raw PDF file content.
        supplier_id : str
            UUID of the supplier in the ``suppliers`` table.
        supplier_terms : str | None
            Default payment terms for this supplier (e.g. "30 dias").
        max_pages : int | None
            Limit how many PDF pages to parse. ``None`` = all pages.
        """

        # 1. Create document record ──────────────────────────
        doc_id = await self.writer.create_document(
            tenant_id=self.tenant_id,
            supplier_id=supplier_id,
            document_type="price_list",
        )

        try:
            # 2. Parse PDF → ExtractedLineItems ──────────────
            #    PdfLlmParser.parse() is synchronous (uses sync OpenAI
            #    client), so run it in a thread to avoid blocking the
            #    async event loop.
            parser = PdfLlmParser(max_pages=max_pages)
            extracted = await asyncio.to_thread(parser.parse, raw_bytes)

            if not extracted:
                await self.writer.update_document_status(doc_id, 0, 0, 0)
                return IngestResult(
                    document_id=doc_id,
                    items_extracted=0,
                    items_matched=0,
                    items_unresolved=0,
                )

            # 3-5. Resolve each item ─────────────────────────
            matched_items: list[dict] = []
            pending_items: list[dict] = []

            for item in extracted:
                try:
                    match = await self.resolver.resolve(
                        raw_description=item.raw_description,
                        supplier_id=supplier_id,
                        supplier_code=item.supplier_code,
                    )
                except Exception:
                    logger.warning(
                        "Resolver failed for '%s', treating as unmatched",
                        item.raw_description[:80],
                        exc_info=True,
                    )
                    # Treat resolver errors as no-match rather than
                    # crashing the whole pipeline.
                    match = _NO_MATCH

                price = item.best_price

                if match.confidence >= CONFIDENCE_AUTO_MATCH and match.product_id:
                    # ── High confidence → store price + create alias ──
                    if price is not None:
                        await self.writer.write_price_record(
                            tenant_id=self.tenant_id,
                            product_id=match.product_id,
                            supplier_id=supplier_id,
                            price=price,
                            terms=supplier_terms,
                            document_id=doc_id,
                        )

                    # Auto-create alias so next upload is instant (exact alias match)
                    alias_created = await self.writer.create_alias(
                        tenant_id=self.tenant_id,
                        product_id=match.product_id,
                        supplier_id=supplier_id,
                        raw_description=item.raw_description,
                        supplier_code=item.supplier_code,
                    )
                    if alias_created:
                        # Generate embedding for the new alias so it
                        # participates in future semantic searches
                        await self.resolver.ensure_embeddings()

                    matched_items.append({
                        "raw_description": item.raw_description,
                        "matched_product": match.product_name,
                        "product_id": match.product_id,
                        "price": price,
                        "confidence": match.confidence,
                        "strategy": match.strategy,
                    })
                else:
                    # ── Low / no confidence → pending match ──
                    suggested_id = (
                        match.product_id
                        if match.confidence >= CONFIDENCE_SUGGEST
                        else None
                    )
                    await self.writer.write_pending_match(
                        tenant_id=self.tenant_id,
                        document_id=doc_id,
                        supplier_id=supplier_id,
                        raw_description=item.raw_description,
                        supplier_code=item.supplier_code,
                        price=price,
                        terms=supplier_terms,
                        suggested_product_id=suggested_id,
                        confidence=match.confidence,
                        strategy=match.strategy,
                    )
                    pending_items.append({
                        "raw_description": item.raw_description,
                        "supplier_code": item.supplier_code,
                        "price": price,
                        "suggested_product": match.product_name if suggested_id else None,
                        "confidence": match.confidence,
                    })

            # 6. Update document stats ───────────────────────
            await self.writer.update_document_status(
                doc_id,
                items_extracted=len(extracted),
                items_matched=len(matched_items),
                items_unresolved=len(pending_items),
            )

            return IngestResult(
                document_id=doc_id,
                items_extracted=len(extracted),
                items_matched=len(matched_items),
                items_unresolved=len(pending_items),
                matched_items=matched_items,
                pending_items=pending_items,
            )

        except Exception as exc:
            # Mark the document as failed so it's visible in the dashboard.
            # The session may be in a failed state, so rollback first.
            logger.error("Ingestion pipeline failed for doc %s: %s", doc_id, exc, exc_info=True)
            try:
                await self.session.rollback()
                await self.writer.mark_document_error(doc_id, str(exc)[:2000])
                await self.session.commit()
            except Exception:
                logger.warning("Could not mark document %s as errored", doc_id, exc_info=True)
            raise


# Sentinel used when the resolver raises an exception on a single item
from app.services.ingestion.product_resolver import MatchResult as _MR
_NO_MATCH = _MR(product_id=None, product_name=None, confidence=0.0, strategy="none")
