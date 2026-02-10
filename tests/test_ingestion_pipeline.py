"""
Ingestion pipeline integration tests (pytest).

Tests the full flow: PDF upload → parse → resolve → DB writes,
plus unit-level checks on the writer and controller components.

Prerequisite:
  1. python -m scripts.migrate_add_embeddings   (one-time pgvector setup)
  2. python -m scripts.seed_from_fixtures        (parse PDFs, seed DB)

Run:
  pytest tests/test_ingestion_pipeline.py -v
"""

import os

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import async_session, set_tenant_context
from app.services.ingestion.controller import (
    IngestController,
    CONFIDENCE_AUTO_MATCH,
    CONFIDENCE_SUGGEST,
)
from app.services.ingestion.writer import IngestWriter, _STRATEGY_TO_DB

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "test_fixtures")


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def openai_client():
    """Module-scoped AsyncOpenAI client."""
    assert settings.openai_api_key, (
        "OPENAI_API_KEY not set in .env — "
        "required for integration tests that call the API"
    )
    return AsyncOpenAI(api_key=settings.openai_api_key)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def db_context():
    """
    Module-scoped context providing tenant / supplier / product IDs
    discovered from the seeded database.

    Also performs pre- and post-cleanup of test-generated aliases.
    """
    async with async_session() as session:
        # ── Tenant ───────────────────────────────────────────
        row = await session.execute(sql_text(
            "SELECT id FROM tenants WHERE name = 'Nova Safra Dev' LIMIT 1"
        ))
        tenant = row.mappings().first()
        assert tenant, (
            "Tenant 'Nova Safra Dev' not found. "
            "Run: python -m scripts.seed_from_fixtures"
        )
        tenant_id = str(tenant["id"])

        # ── Suppliers ────────────────────────────────────────
        supplier_ids: dict[str, str] = {}
        supplier_terms: dict[str, str] = {}
        for name in ["JRG Alimentos", "Ibérica Comércio"]:
            row = await session.execute(sql_text(
                "SELECT id, default_terms FROM suppliers WHERE name = :n LIMIT 1"
            ), {"n": name})
            s = row.mappings().first()
            assert s, f"Supplier '{name}' not found"
            supplier_ids[name] = str(s["id"])
            supplier_terms[name] = s["default_terms"] or "30 dias"

        # ── Sample product (for writer unit tests) ───────────
        row = await session.execute(sql_text("SELECT id FROM products LIMIT 1"))
        p = row.mappings().first()
        assert p, "No products seeded"

        # ── Pre-cleanup ──────────────────────────────────────
        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE source = 'price_list'"
        ))
        await session.commit()

        yield {
            "tenant_id": tenant_id,
            "supplier_ids": supplier_ids,
            "supplier_terms": supplier_terms,
            "product_id": str(p["id"]),
        }

    # ── Post-cleanup: remove any test aliases left behind ────
    async with async_session() as session:
        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE source = 'price_list'"
        ))
        await session.commit()


@pytest_asyncio.fixture(loop_scope="module")
async def session(db_context):
    """Function-scoped async DB session with tenant context pre-set."""
    async with async_session() as s:
        await set_tenant_context(s, db_context["tenant_id"])
        yield s


@pytest.fixture(scope="module")
def jrg_pdf_bytes():
    """Raw bytes of the JRG Alimentos price list PDF."""
    path = os.path.join(FIXTURES_DIR, "price_list_jrg.pdf")
    if not os.path.exists(path):
        pytest.skip("price_list_jrg.pdf not found in test_fixtures/")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def iberica_pdf_bytes():
    """Raw bytes of the Ibérica Comércio price list PDF."""
    path = os.path.join(FIXTURES_DIR, "price_list_iberica.pdf")
    if not os.path.exists(path):
        pytest.skip("price_list_iberica.pdf not found in test_fixtures/")
    with open(path, "rb") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════
# 1. Strategy mapping  (sync, no DB needed)
# ═══════════════════════════════════════════════════════════════

class TestStrategyMapping:
    """Verify resolver strategy names map to valid DB enum values."""

    ALLOWED_DB_VALUES = {"fuzzy_text", "embedding", "none"}

    @pytest.mark.parametrize("strategy,expected_db", [
        ("exact_code",   "fuzzy_text"),
        ("exact_alias",  "fuzzy_text"),
        ("semantic",     "embedding"),
        ("llm_verified", "embedding"),
        ("none",         "none"),
    ])
    def test_maps_to_correct_db_value(self, strategy, expected_db):
        assert _STRATEGY_TO_DB[strategy] == expected_db
        assert expected_db in self.ALLOWED_DB_VALUES

    def test_unknown_strategy_defaults_to_none(self):
        assert _STRATEGY_TO_DB.get("something_unknown", "none") == "none"


# ═══════════════════════════════════════════════════════════════
# 2. Writer — document lifecycle
# ═══════════════════════════════════════════════════════════════

class TestWriterDocumentLifecycle:
    """Test create → update → error cycle for documents."""

    async def test_create_document(self, session, db_context):
        writer = IngestWriter(session)
        sid = db_context["supplier_ids"]["JRG Alimentos"]

        doc_id = await writer.create_document(
            tenant_id=db_context["tenant_id"],
            supplier_id=sid,
            document_type="price_list",
            subject="pytest lifecycle test",
        )
        assert len(doc_id) == 36

        row = await session.execute(sql_text(
            "SELECT processing_status, subject FROM documents WHERE id = :did"
        ), {"did": doc_id})
        doc = row.mappings().first()
        assert doc["processing_status"] == "processing"
        assert doc["subject"] == "pytest lifecycle test"

        await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": doc_id})
        await session.commit()

    async def test_update_document_status(self, session, db_context):
        writer = IngestWriter(session)
        sid = db_context["supplier_ids"]["JRG Alimentos"]

        doc_id = await writer.create_document(
            tenant_id=db_context["tenant_id"], supplier_id=sid,
        )
        await writer.update_document_status(
            doc_id, items_extracted=50, items_matched=30, items_unresolved=20,
        )

        row = await session.execute(sql_text(
            "SELECT processing_status, items_extracted, items_matched, "
            "items_unresolved, processed_at "
            "FROM documents WHERE id = :did"
        ), {"did": doc_id})
        doc = row.mappings().first()
        assert doc["processing_status"] == "processed"
        assert doc["items_extracted"] == 50
        assert doc["items_matched"] == 30
        assert doc["items_unresolved"] == 20
        assert doc["processed_at"] is not None

        await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": doc_id})
        await session.commit()

    async def test_mark_document_error(self, session, db_context):
        writer = IngestWriter(session)
        sid = db_context["supplier_ids"]["JRG Alimentos"]

        doc_id = await writer.create_document(
            tenant_id=db_context["tenant_id"], supplier_id=sid,
        )
        await writer.mark_document_error(doc_id, "Something went wrong")

        row = await session.execute(sql_text(
            "SELECT processing_status, error_detail "
            "FROM documents WHERE id = :did"
        ), {"did": doc_id})
        doc = row.mappings().first()
        assert doc["processing_status"] == "failed"
        assert doc["error_detail"] == "Something went wrong"

        await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": doc_id})
        await session.commit()


# ═══════════════════════════════════════════════════════════════
# 3. Writer — price records, pending matches, aliases
# ═══════════════════════════════════════════════════════════════

class TestWriterRecords:
    """Test DB writes for price records, pending matches, and aliases."""

    async def test_write_price_record(self, session, db_context):
        writer = IngestWriter(session)
        tid = db_context["tenant_id"]
        sid = db_context["supplier_ids"]["JRG Alimentos"]
        pid = db_context["product_id"]

        doc_id = await writer.create_document(tenant_id=tid, supplier_id=sid)
        await writer.write_price_record(
            tenant_id=tid, product_id=pid, supplier_id=sid,
            price=42.50, terms="30 dias", document_id=doc_id,
        )

        row = await session.execute(sql_text("""
            SELECT unit_price, payment_terms, source_type, source_document_id::text
            FROM price_records WHERE source_document_id = :did
        """), {"did": doc_id})
        pr = row.mappings().first()
        assert pr is not None
        assert float(pr["unit_price"]) == 42.50
        assert pr["payment_terms"] == "30 dias"
        assert pr["source_type"] == "price_list"
        assert pr["source_document_id"] == doc_id

        await session.execute(sql_text(
            "DELETE FROM price_records WHERE source_document_id = :did"
        ), {"did": doc_id})
        await session.execute(sql_text(
            "DELETE FROM documents WHERE id = :did"
        ), {"did": doc_id})
        await session.commit()

    async def test_pending_match_maps_llm_verified_to_embedding(self, session, db_context):
        writer = IngestWriter(session)
        tid = db_context["tenant_id"]
        sid = db_context["supplier_ids"]["JRG Alimentos"]
        pid = db_context["product_id"]

        doc_id = await writer.create_document(tenant_id=tid, supplier_id=sid)
        await writer.write_pending_match(
            tenant_id=tid, document_id=doc_id, supplier_id=sid,
            raw_description="TEST PRODUCT 5KG",
            supplier_code="99999",
            price=15.00, terms="30 dias",
            suggested_product_id=pid,
            confidence=0.65,
            strategy="llm_verified",
        )

        row = await session.execute(sql_text("""
            SELECT match_strategy, confidence, status, suggested_product_id::text
            FROM pending_matches WHERE document_id = :did
        """), {"did": doc_id})
        pm = row.mappings().first()
        assert pm is not None
        assert pm["match_strategy"] == "embedding"  # llm_verified → embedding
        assert pm["status"] == "pending"
        assert pm["suggested_product_id"] == pid

        await session.execute(sql_text(
            "DELETE FROM pending_matches WHERE document_id = :did"
        ), {"did": doc_id})
        await session.execute(sql_text(
            "DELETE FROM documents WHERE id = :did"
        ), {"did": doc_id})
        await session.commit()

    async def test_pending_match_none_strategy(self, session, db_context):
        writer = IngestWriter(session)
        tid = db_context["tenant_id"]
        sid = db_context["supplier_ids"]["JRG Alimentos"]

        doc_id = await writer.create_document(tenant_id=tid, supplier_id=sid)
        await writer.write_pending_match(
            tenant_id=tid, document_id=doc_id, supplier_id=sid,
            raw_description="TOTALLY UNKNOWN",
            supplier_code=None,
            price=None, terms=None,
            suggested_product_id=None,
            confidence=0.0,
            strategy="none",
        )

        row = await session.execute(sql_text("""
            SELECT match_strategy, suggested_product_id
            FROM pending_matches WHERE document_id = :did
        """), {"did": doc_id})
        pm = row.mappings().first()
        assert pm["match_strategy"] == "none"
        assert pm["suggested_product_id"] is None

        await session.execute(sql_text(
            "DELETE FROM pending_matches WHERE document_id = :did"
        ), {"did": doc_id})
        await session.execute(sql_text(
            "DELETE FROM documents WHERE id = :did"
        ), {"did": doc_id})
        await session.commit()

    async def test_create_alias_returns_true(self, session, db_context):
        writer = IngestWriter(session)
        tid = db_context["tenant_id"]
        sid = db_context["supplier_ids"]["JRG Alimentos"]
        pid = db_context["product_id"]

        created = await writer.create_alias(
            tenant_id=tid, product_id=pid, supplier_id=sid,
            raw_description="UNIQUE_PYTEST_ALIAS_XYZ",
            supplier_code="88888",
        )
        assert created is True

        row = await session.execute(sql_text("""
            SELECT source, confirmed, normalised_description
            FROM product_aliases
            WHERE raw_description = 'UNIQUE_PYTEST_ALIAS_XYZ'
        """))
        alias = row.mappings().first()
        assert alias is not None
        assert alias["source"] == "price_list"
        assert alias["confirmed"] is True
        assert alias["normalised_description"] == "unique_pytest_alias_xyz"

        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE raw_description = 'UNIQUE_PYTEST_ALIAS_XYZ'"
        ))
        await session.commit()

    async def test_duplicate_alias_returns_false(self, session, db_context):
        writer = IngestWriter(session)
        tid = db_context["tenant_id"]
        sid = db_context["supplier_ids"]["JRG Alimentos"]
        pid = db_context["product_id"]

        # Create first
        await writer.create_alias(
            tenant_id=tid, product_id=pid, supplier_id=sid,
            raw_description="PYTEST_DUP_ALIAS",
            supplier_code=None,
        )

        # Duplicate should return False
        created = await writer.create_alias(
            tenant_id=tid, product_id=pid, supplier_id=sid,
            raw_description="PYTEST_DUP_ALIAS",
            supplier_code=None,
        )
        assert created is False

        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE raw_description = 'PYTEST_DUP_ALIAS'"
        ))
        await session.commit()


# ═══════════════════════════════════════════════════════════════
# 4. Full pipeline — JRG (1 page)
#
#    NOTE: This test intentionally keeps auto-created aliases in
#    the DB so the re-ingestion test (5) can verify they speed
#    up matching.  The module-scoped db_context fixture handles
#    final alias cleanup.
# ═══════════════════════════════════════════════════════════════

async def test_full_pipeline_jrg(session, db_context, openai_client, jrg_pdf_bytes):
    """End-to-end: parse JRG page 1 → resolve → write to DB."""
    controller = IngestController(
        session, db_context["tenant_id"], openai_client,
    )
    result = await controller.ingest_price_list(
        raw_bytes=jrg_pdf_bytes,
        supplier_id=db_context["supplier_ids"]["JRG Alimentos"],
        supplier_terms=db_context["supplier_terms"]["JRG Alimentos"],
        max_pages=1,
    )

    # ── Result structure ─────────────────────────────────────
    assert len(result.document_id) == 36
    assert result.items_extracted > 0
    assert result.items_matched + result.items_unresolved == result.items_extracted
    assert result.items_matched > 0
    assert len(result.matched_items) == result.items_matched
    assert len(result.pending_items) == result.items_unresolved

    # ── Matched items ────────────────────────────────────────
    required_matched_keys = {
        "raw_description", "matched_product", "product_id",
        "price", "confidence", "strategy",
    }
    for m in result.matched_items:
        assert required_matched_keys <= m.keys()
        assert m["confidence"] >= CONFIDENCE_AUTO_MATCH
        assert m["product_id"] is not None
        assert m["matched_product"] is not None

    # ── Pending items ────────────────────────────────────────
    required_pending_keys = {
        "raw_description", "supplier_code", "price",
        "suggested_product", "confidence",
    }
    for p in result.pending_items:
        assert required_pending_keys <= p.keys()
        assert p["confidence"] < CONFIDENCE_AUTO_MATCH

    # ── DB: document record ──────────────────────────────────
    row = await session.execute(sql_text(
        "SELECT processing_status, items_extracted, items_matched, "
        "items_unresolved, processed_at "
        "FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    doc = row.mappings().first()
    assert doc["processing_status"] == "processed"
    assert doc["items_extracted"] == result.items_extracted
    assert doc["items_matched"] == result.items_matched
    assert doc["items_unresolved"] == result.items_unresolved
    assert doc["processed_at"] is not None

    # ── DB: price records ────────────────────────────────────
    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    items_with_price = sum(1 for m in result.matched_items if m["price"] is not None)
    assert row.scalar() == items_with_price

    # ── DB: pending matches ──────────────────────────────────
    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    assert row.scalar() == result.items_unresolved

    # ── DB: pending match strategies are valid ───────────────
    row = await session.execute(sql_text(
        "SELECT DISTINCT match_strategy FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    db_strategies = {r[0] for r in row.all()}
    assert db_strategies <= {"fuzzy_text", "embedding", "none"}

    # ── DB: auto-created aliases ─────────────────────────────
    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM product_aliases "
        "WHERE source = 'price_list' AND supplier_id = :sid"
    ), {"sid": db_context["supplier_ids"]["JRG Alimentos"]})
    assert row.scalar() >= 1

    # ── Cleanup (keep aliases for re-ingestion test) ─────────
    await session.execute(sql_text(
        "DELETE FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    await session.commit()


# ═══════════════════════════════════════════════════════════════
# 5. Re-ingestion — aliases speed up matching
#
#    Runs AFTER test 4.  Since test 4 left aliases in the DB,
#    this run should find some items via exact_alias.
# ═══════════════════════════════════════════════════════════════

async def test_reingestion_uses_aliases(session, db_context, openai_client, jrg_pdf_bytes):
    """Re-ingesting the same PDF should match more items via exact_alias."""
    controller = IngestController(
        session, db_context["tenant_id"], openai_client,
    )
    result = await controller.ingest_price_list(
        raw_bytes=jrg_pdf_bytes,
        supplier_id=db_context["supplier_ids"]["JRG Alimentos"],
        supplier_terms=db_context["supplier_terms"]["JRG Alimentos"],
        max_pages=1,
    )

    exact_alias_count = sum(
        1 for m in result.matched_items if m["strategy"] == "exact_alias"
    )
    assert exact_alias_count > 0, (
        "Expected auto-created aliases from previous test to produce "
        "exact_alias matches on re-ingestion"
    )
    assert result.items_matched > 0
    assert result.items_matched / result.items_extracted >= 0.30

    # ── Cleanup ──────────────────────────────────────────────
    await session.execute(sql_text(
        "DELETE FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    await session.commit()


# ═══════════════════════════════════════════════════════════════
# 6. Full pipeline — Ibérica (1 page)
# ═══════════════════════════════════════════════════════════════

async def test_full_pipeline_iberica(session, db_context, openai_client, iberica_pdf_bytes):
    """End-to-end: parse Ibérica page 1 → resolve → write to DB."""
    controller = IngestController(
        session, db_context["tenant_id"], openai_client,
    )
    result = await controller.ingest_price_list(
        raw_bytes=iberica_pdf_bytes,
        supplier_id=db_context["supplier_ids"]["Ibérica Comércio"],
        supplier_terms=db_context["supplier_terms"]["Ibérica Comércio"],
        max_pages=1,
    )

    assert result.items_extracted > 0
    assert result.items_matched + result.items_unresolved == result.items_extracted
    assert result.items_matched > 0

    # DB verification
    row = await session.execute(sql_text(
        "SELECT processing_status FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    assert row.mappings().first()["processing_status"] == "processed"

    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    assert row.scalar() >= 1

    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    assert row.scalar() == result.items_unresolved

    # ── Cleanup ──────────────────────────────────────────────
    await session.execute(sql_text(
        "DELETE FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    await session.commit()


# ═══════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Verify the pipeline handles bad input gracefully."""

    async def test_empty_bytes_raises(self, session, db_context, openai_client):
        controller = IngestController(
            session, db_context["tenant_id"], openai_client,
        )
        with pytest.raises(Exception):
            await controller.ingest_price_list(
                raw_bytes=b"",
                supplier_id=db_context["supplier_ids"]["JRG Alimentos"],
                max_pages=1,
            )
        # Cleanup failed document
        await session.execute(sql_text(
            "DELETE FROM documents WHERE processing_status = 'failed'"
        ))
        await session.commit()

    async def test_non_pdf_bytes_raises(self, session, db_context, openai_client):
        controller = IngestController(
            session, db_context["tenant_id"], openai_client,
        )
        with pytest.raises(Exception):
            await controller.ingest_price_list(
                raw_bytes=b"This is definitely not a PDF file.",
                supplier_id=db_context["supplier_ids"]["JRG Alimentos"],
                max_pages=1,
            )
        await session.execute(sql_text(
            "DELETE FROM documents WHERE processing_status = 'failed'"
        ))
        await session.commit()
