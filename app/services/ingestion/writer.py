"""
Database writer for the ingestion pipeline.

Provides focused INSERT/UPDATE methods for each table touched during
price list ingestion: documents, price_records, pending_matches,
product_aliases.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from uuid import uuid4

# The pending_matches table has a CHECK constraint limiting match_strategy
# to these values.  Map resolver strategy names → DB-allowed values.
_STRATEGY_TO_DB: dict[str, str] = {
    "exact_code":   "fuzzy_text",   # closest analog in the DB enum
    "exact_alias":  "fuzzy_text",
    "semantic":     "embedding",
    "llm_verified": "embedding",
    "none":         "none",
}


class IngestWriter:
    """Writes ingestion results to the database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Documents ─────────────────────────────────────────

    async def create_document(
        self,
        tenant_id: str,
        supplier_id: str,
        document_type: str = "price_list",
        subject: str | None = None,
    ) -> str:
        """Create a document record in 'processing' state.  Returns the new ID."""
        doc_id = str(uuid4())
        await self.session.execute(text("""
            INSERT INTO documents (id, tenant_id, supplier_id, document_type, subject, processing_status)
            VALUES (:id, :tid, :sid, :dtype, :subject, 'processing')
        """), {
            "id": doc_id, "tid": tenant_id, "sid": supplier_id,
            "dtype": document_type, "subject": subject,
        })
        return doc_id

    async def update_document_status(
        self,
        document_id: str,
        items_extracted: int,
        items_matched: int,
        items_unresolved: int,
    ) -> None:
        """Mark a document as processed with final counts."""
        await self.session.execute(text("""
            UPDATE documents
            SET items_extracted  = :ext,
                items_matched    = :mat,
                items_unresolved = :unr,
                processing_status = 'processed',
                processed_at      = NOW()
            WHERE id = :did
        """), {
            "did": document_id,
            "ext": items_extracted,
            "mat": items_matched,
            "unr": items_unresolved,
        })

    async def mark_document_error(
        self, document_id: str, error_detail: str,
    ) -> None:
        """Mark a document as failed."""
        await self.session.execute(text("""
            UPDATE documents
            SET processing_status = 'failed',
                error_detail      = :err,
                processed_at      = NOW()
            WHERE id = :did
        """), {"did": document_id, "err": error_detail})

    # ── Price records ─────────────────────────────────────

    async def write_price_record(
        self,
        tenant_id: str,
        product_id: str,
        supplier_id: str,
        price: float,
        terms: str | None,
        document_id: str,
    ) -> None:
        """Insert a new price observation from a price list."""
        await self.session.execute(text("""
            INSERT INTO price_records
                (tenant_id, product_id, supplier_id, unit_price,
                 payment_terms, source_type, source_document_id)
            VALUES (:tid, :pid, :sid, :price, :terms, 'price_list', :did)
        """), {
            "tid": tenant_id, "pid": product_id, "sid": supplier_id,
            "price": price, "terms": terms, "did": document_id,
        })

    # ── Pending matches ───────────────────────────────────

    async def write_pending_match(
        self,
        tenant_id: str,
        document_id: str,
        supplier_id: str,
        raw_description: str,
        supplier_code: str | None,
        price: float | None,
        terms: str | None,
        suggested_product_id: str | None,
        confidence: float,
        strategy: str,
    ) -> None:
        """Queue an unresolved item for buyer review."""
        db_strategy = _STRATEGY_TO_DB.get(strategy, "none")
        await self.session.execute(text("""
            INSERT INTO pending_matches
                (tenant_id, document_id, supplier_id, raw_description,
                 supplier_code, extracted_price, extracted_terms,
                 suggested_product_id, confidence, match_strategy)
            VALUES (:tid, :did, :sid, :raw, :code,
                    :price, :terms, :sug_pid, :conf, :strat)
        """), {
            "tid": tenant_id, "did": document_id, "sid": supplier_id,
            "raw": raw_description, "code": supplier_code,
            "price": price, "terms": terms,
            "sug_pid": suggested_product_id, "conf": confidence,
            "strat": db_strategy,
        })

    # ── Product aliases ───────────────────────────────────

    async def create_alias(
        self,
        tenant_id: str,
        product_id: str,
        supplier_id: str,
        raw_description: str,
        supplier_code: str | None,
    ) -> bool:
        """
        Auto-create a confirmed product alias for a high-confidence match.

        Next time the same supplier sends this description, the resolver
        will find it via exact alias lookup (stage 2) instead of the
        slower embedding + LLM path.

        Returns True if a new alias was created, False if it already existed.
        """
        # Check if an alias with this exact description already exists
        existing = await self.session.execute(text("""
            SELECT id FROM product_aliases
            WHERE raw_description = :raw AND supplier_id = :sid
            LIMIT 1
        """), {"raw": raw_description, "sid": supplier_id})

        if existing.first() is not None:
            return False

        await self.session.execute(text("""
            INSERT INTO product_aliases
                (tenant_id, product_id, supplier_id, raw_description,
                 normalised_description, supplier_code, source, confirmed)
            VALUES (:tid, :pid, :sid, :raw, :norm, :code, 'price_list', true)
        """), {
            "tid": tenant_id, "pid": product_id, "sid": supplier_id,
            "raw": raw_description, "norm": raw_description.lower().strip(),
            "code": supplier_code,
        })
        return True
