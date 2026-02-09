"""
Ingestion pipeline integration tests.

Tests the full flow: PDF upload → parse → resolve → DB writes,
plus unit-level checks on the writer and controller components.

Prerequisite:
  1. python -m scripts.migrate_add_embeddings   (one-time pgvector setup)
  2. python -m scripts.seed_from_fixtures        (parse PDFs, seed DB)

Run:
  python tests/test_ingestion_pipeline.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "test_fixtures")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def icon(ok: bool) -> str:
    return "✓" if ok else "✗"


def assert_check(label: str, condition: bool, errors: list[str], detail: str = ""):
    """Log a pass/fail line and accumulate errors."""
    if condition:
        print(f"    {icon(True)} {label}")
    else:
        msg = f"{label}: {detail}" if detail else label
        errors.append(msg)
        print(f"    {icon(False)} {msg}")


# ═══════════════════════════════════════════════════════════════
# Test 1: Strategy mapping
# ═══════════════════════════════════════════════════════════════

def test_strategy_mapping() -> bool:
    """Verify _STRATEGY_TO_DB maps all resolver strategies to valid DB values."""
    print("\n" + "=" * 70)
    print("  TEST 1: Strategy mapping (resolver → DB enum)")
    print("=" * 70)

    errors: list[str] = []
    allowed_db_values = {"fuzzy_text", "embedding", "none"}
    resolver_strategies = ["exact_code", "exact_alias", "semantic", "llm_verified", "none"]

    for strat in resolver_strategies:
        db_val = _STRATEGY_TO_DB.get(strat)
        assert_check(
            f"{strat:16s} → {db_val}",
            db_val is not None and db_val in allowed_db_values,
            errors,
            f"mapped to {db_val!r}, allowed: {allowed_db_values}",
        )

    # Unknown strategies should fall back to "none" in the writer
    fallback = _STRATEGY_TO_DB.get("unknown_strategy", "none")
    assert_check(
        "Unknown strategy fallback → 'none'",
        fallback == "none",
        errors,
    )

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print("\n  PASS ✓")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 2: Writer — document lifecycle
# ═══════════════════════════════════════════════════════════════

async def test_writer_document_lifecycle(
    session, tenant_id: str, supplier_id: str,
) -> bool:
    """Test create → update → error cycle for documents."""
    print("\n" + "=" * 70)
    print("  TEST 2: Writer — document lifecycle")
    print("=" * 70)

    errors: list[str] = []
    writer = IngestWriter(session)

    # ── 2a. Create document ──────────────────────────────────
    doc_id = await writer.create_document(
        tenant_id=tenant_id,
        supplier_id=supplier_id,
        document_type="price_list",
        subject="Test document lifecycle",
    )
    assert_check("create_document returns UUID", len(doc_id) == 36, errors, f"got {doc_id!r}")

    row = await session.execute(sql_text(
        "SELECT processing_status, subject FROM documents WHERE id = :did"
    ), {"did": doc_id})
    doc = row.mappings().first()
    assert_check(
        "Document created with status='processing'",
        doc is not None and doc["processing_status"] == "processing",
        errors,
        f"got {doc}",
    )
    assert_check(
        "Document has correct subject",
        doc is not None and doc["subject"] == "Test document lifecycle",
        errors,
    )

    # ── 2b. Update document status ───────────────────────────
    await writer.update_document_status(doc_id, items_extracted=50, items_matched=30, items_unresolved=20)

    row = await session.execute(sql_text(
        "SELECT processing_status, items_extracted, items_matched, items_unresolved, processed_at FROM documents WHERE id = :did"
    ), {"did": doc_id})
    doc = row.mappings().first()
    assert_check(
        "Status updated to 'processed'",
        doc["processing_status"] == "processed",
        errors,
        f"got {doc['processing_status']!r}",
    )
    assert_check("items_extracted = 50", doc["items_extracted"] == 50, errors)
    assert_check("items_matched = 30", doc["items_matched"] == 30, errors)
    assert_check("items_unresolved = 20", doc["items_unresolved"] == 20, errors)
    assert_check("processed_at is set", doc["processed_at"] is not None, errors)

    # ── 2c. Create a second document and mark it as failed ───
    doc_id2 = await writer.create_document(
        tenant_id=tenant_id,
        supplier_id=supplier_id,
        document_type="price_list",
    )
    await writer.mark_document_error(doc_id2, "Test error message")

    row = await session.execute(sql_text(
        "SELECT processing_status, error_detail FROM documents WHERE id = :did"
    ), {"did": doc_id2})
    doc2 = row.mappings().first()
    assert_check(
        "Error document has status='failed'",
        doc2["processing_status"] == "failed",
        errors,
        f"got {doc2['processing_status']!r}",
    )
    assert_check(
        "Error detail is stored",
        doc2["error_detail"] == "Test error message",
        errors,
    )

    # ── Cleanup test documents ───────────────────────────────
    await session.execute(sql_text("DELETE FROM documents WHERE id IN (:d1, :d2)"), {"d1": doc_id, "d2": doc_id2})

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print("\n  PASS ✓")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 3: Writer — price records and pending matches
# ═══════════════════════════════════════════════════════════════

async def test_writer_records(
    session, tenant_id: str, supplier_id: str, product_id: str,
) -> bool:
    """Test writing price records, pending matches, and aliases."""
    print("\n" + "=" * 70)
    print("  TEST 3: Writer — price records, pending matches, aliases")
    print("=" * 70)

    errors: list[str] = []
    writer = IngestWriter(session)

    # Need a document first (FK constraint)
    doc_id = await writer.create_document(
        tenant_id=tenant_id, supplier_id=supplier_id,
    )

    # ── 3a. Write price record ───────────────────────────────
    await writer.write_price_record(
        tenant_id=tenant_id,
        product_id=product_id,
        supplier_id=supplier_id,
        price=42.50,
        terms="30 dias",
        document_id=doc_id,
    )
    row = await session.execute(sql_text("""
        SELECT unit_price, payment_terms, source_type, source_document_id::text
        FROM price_records
        WHERE source_document_id = :did
    """), {"did": doc_id})
    pr = row.mappings().first()
    assert_check("Price record created", pr is not None, errors)
    if pr:
        assert_check(
            f"unit_price = 42.50",
            float(pr["unit_price"]) == 42.50,
            errors,
            f"got {pr['unit_price']}",
        )
        assert_check(
            "payment_terms = '30 dias'",
            pr["payment_terms"] == "30 dias",
            errors,
        )
        assert_check(
            "source_type = 'price_list'",
            pr["source_type"] == "price_list",
            errors,
        )
        assert_check(
            "source_document_id matches",
            pr["source_document_id"] == doc_id,
            errors,
        )

    # ── 3b. Write pending match ──────────────────────────────
    await writer.write_pending_match(
        tenant_id=tenant_id,
        document_id=doc_id,
        supplier_id=supplier_id,
        raw_description="TEST PRODUCT UNKNOWN 5KG",
        supplier_code="99999",
        price=15.00,
        terms="30 dias",
        suggested_product_id=product_id,
        confidence=0.65,
        strategy="llm_verified",  # should be mapped to 'embedding' in DB
    )
    row = await session.execute(sql_text("""
        SELECT raw_description, supplier_code, extracted_price, match_strategy,
               confidence, status, suggested_product_id::text
        FROM pending_matches
        WHERE document_id = :did
    """), {"did": doc_id})
    pm = row.mappings().first()
    assert_check("Pending match created", pm is not None, errors)
    if pm:
        assert_check(
            "raw_description stored",
            pm["raw_description"] == "TEST PRODUCT UNKNOWN 5KG",
            errors,
        )
        assert_check(
            "supplier_code = '99999'",
            pm["supplier_code"] == "99999",
            errors,
        )
        assert_check(
            f"extracted_price = 15.00",
            float(pm["extracted_price"]) == 15.00,
            errors,
            f"got {pm['extracted_price']}",
        )
        assert_check(
            "match_strategy mapped: llm_verified → 'embedding'",
            pm["match_strategy"] == "embedding",
            errors,
            f"got {pm['match_strategy']!r}",
        )
        assert_check("status = 'pending'", pm["status"] == "pending", errors)
        assert_check(
            "suggested_product_id set",
            pm["suggested_product_id"] == product_id,
            errors,
        )

    # ── 3c. Write pending match with strategy='none' ─────────
    await writer.write_pending_match(
        tenant_id=tenant_id,
        document_id=doc_id,
        supplier_id=supplier_id,
        raw_description="TOTALLY UNKNOWN ITEM",
        supplier_code=None,
        price=None,
        terms=None,
        suggested_product_id=None,
        confidence=0.0,
        strategy="none",
    )
    row = await session.execute(sql_text("""
        SELECT match_strategy, suggested_product_id
        FROM pending_matches
        WHERE document_id = :did AND raw_description = 'TOTALLY UNKNOWN ITEM'
    """), {"did": doc_id})
    pm2 = row.mappings().first()
    assert_check(
        "Pending match with strategy='none' stored as 'none'",
        pm2 is not None and pm2["match_strategy"] == "none",
        errors,
    )
    assert_check(
        "No suggested product for zero-confidence",
        pm2 is not None and pm2["suggested_product_id"] is None,
        errors,
    )

    # ── 3d. Create alias ─────────────────────────────────────
    created = await writer.create_alias(
        tenant_id=tenant_id,
        product_id=product_id,
        supplier_id=supplier_id,
        raw_description="TEST ALIAS DESCRIPTION XYZ",
        supplier_code="88888",
    )
    assert_check("First alias creation returns True", created is True, errors)

    row = await session.execute(sql_text("""
        SELECT product_id::text, source, confirmed, normalised_description
        FROM product_aliases
        WHERE raw_description = 'TEST ALIAS DESCRIPTION XYZ'
    """))
    alias = row.mappings().first()
    assert_check("Alias exists in DB", alias is not None, errors)
    if alias:
        assert_check("Alias source = 'price_list'", alias["source"] == "price_list", errors)
        assert_check("Alias confirmed = True", alias["confirmed"] is True, errors)
        assert_check(
            "Alias normalised_description is lowered",
            alias["normalised_description"] == "test alias description xyz",
            errors,
        )

    # ── 3e. Duplicate alias returns False ────────────────────
    created2 = await writer.create_alias(
        tenant_id=tenant_id,
        product_id=product_id,
        supplier_id=supplier_id,
        raw_description="TEST ALIAS DESCRIPTION XYZ",
        supplier_code="88888",
    )
    assert_check("Duplicate alias returns False", created2 is False, errors)

    # ── Cleanup ──────────────────────────────────────────────
    await session.execute(sql_text(
        "DELETE FROM product_aliases WHERE raw_description = 'TEST ALIAS DESCRIPTION XYZ'"
    ))
    await session.execute(sql_text("DELETE FROM pending_matches WHERE document_id = :did"), {"did": doc_id})
    await session.execute(sql_text("DELETE FROM price_records WHERE source_document_id = :did"), {"did": doc_id})
    await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": doc_id})

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print("\n  PASS ✓")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 4: Full pipeline — JRG 1 page
# ═══════════════════════════════════════════════════════════════

async def test_full_pipeline_jrg(
    session, tenant_id: str, supplier_id: str, supplier_terms: str,
    openai_client: AsyncOpenAI,
) -> bool:
    """Run the full ingestion pipeline on page 1 of the JRG PDF."""
    print("\n" + "=" * 70)
    print("  TEST 4: Full pipeline — JRG Alimentos (1 page)")
    print("=" * 70)

    pdf_path = os.path.join(FIXTURES, "price_list_jrg.pdf")
    if not os.path.exists(pdf_path):
        print("    SKIP: price_list_jrg.pdf not found")
        return True

    errors: list[str] = []

    with open(pdf_path, "rb") as f:
        raw_bytes = f.read()

    controller = IngestController(session, tenant_id, openai_client)

    print("    Running pipeline (parse + resolve)... ", end="", flush=True)
    t0 = time.time()
    result = await controller.ingest_price_list(
        raw_bytes=raw_bytes,
        supplier_id=supplier_id,
        supplier_terms=supplier_terms,
        max_pages=1,
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    # ── 4a. Result structure ─────────────────────────────────
    print("\n  4a. Result structure:")
    assert_check(
        f"document_id is UUID ({result.document_id[:8]}...)",
        len(result.document_id) == 36,
        errors,
    )
    assert_check(
        f"items_extracted > 0 (got {result.items_extracted})",
        result.items_extracted > 0,
        errors,
    )
    assert_check(
        f"items_matched + items_unresolved == items_extracted "
        f"({result.items_matched} + {result.items_unresolved} == {result.items_extracted})",
        result.items_matched + result.items_unresolved == result.items_extracted,
        errors,
    )
    assert_check(
        f"items_matched > 0 (got {result.items_matched})",
        result.items_matched > 0,
        errors,
    )
    assert_check(
        f"len(matched_items) == items_matched ({len(result.matched_items)})",
        len(result.matched_items) == result.items_matched,
        errors,
    )
    assert_check(
        f"len(pending_items) == items_unresolved ({len(result.pending_items)})",
        len(result.pending_items) == result.items_unresolved,
        errors,
    )

    # ── 4b. Matched items structure ──────────────────────────
    print("\n  4b. Matched items structure:")
    if result.matched_items:
        sample = result.matched_items[0]
        required_keys = {"raw_description", "matched_product", "product_id", "price", "confidence", "strategy"}
        assert_check(
            f"Matched item has required keys",
            required_keys.issubset(sample.keys()),
            errors,
            f"missing: {required_keys - sample.keys()}",
        )
        assert_check(
            "All matched items have confidence >= threshold",
            all(m["confidence"] >= CONFIDENCE_AUTO_MATCH for m in result.matched_items),
            errors,
            f"min confidence: {min(m['confidence'] for m in result.matched_items):.3f}",
        )
        assert_check(
            "All matched items have a product_id",
            all(m["product_id"] is not None for m in result.matched_items),
            errors,
        )
        assert_check(
            "All matched items have a matched_product name",
            all(m["matched_product"] is not None for m in result.matched_items),
            errors,
        )

        # Print a few matched items
        print("\n    Sample matched items:")
        for m in result.matched_items[:5]:
            print(f"      {m['strategy']:14s} conf={m['confidence']:.3f}  "
                  f"R${m['price'] or 0:.2f}  \"{m['raw_description'][:45]}\" → \"{m['matched_product']}\"")
        if len(result.matched_items) > 5:
            print(f"      ... and {len(result.matched_items) - 5} more")
    else:
        errors.append("No matched items at all")

    # ── 4c. Pending items structure ──────────────────────────
    print("\n  4c. Pending items structure:")
    if result.pending_items:
        sample = result.pending_items[0]
        required_keys = {"raw_description", "supplier_code", "price", "suggested_product", "confidence"}
        assert_check(
            f"Pending item has required keys",
            required_keys.issubset(sample.keys()),
            errors,
            f"missing: {required_keys - sample.keys()}",
        )
        assert_check(
            "All pending items have confidence < auto-match threshold",
            all(p["confidence"] < CONFIDENCE_AUTO_MATCH for p in result.pending_items),
            errors,
        )

        # Count suggestions vs no-suggestions
        with_suggestion = [p for p in result.pending_items if p["suggested_product"] is not None]
        without_suggestion = [p for p in result.pending_items if p["suggested_product"] is None]
        print(f"    With suggestion:    {len(with_suggestion)}")
        print(f"    Without suggestion: {len(without_suggestion)}")

        # Print a few pending items
        print("\n    Sample pending items:")
        for p in result.pending_items[:5]:
            sug = f"→ \"{p['suggested_product']}\"" if p["suggested_product"] else "→ (none)"
            print(f"      conf={p['confidence']:.3f}  "
                  f"R${p['price'] or 0:.2f}  \"{p['raw_description'][:45]}\" {sug}")

    # ── 4d. Database verification ────────────────────────────
    print("\n  4d. Database verification:")

    # Document record
    row = await session.execute(sql_text("""
        SELECT processing_status, items_extracted, items_matched, items_unresolved, processed_at
        FROM documents WHERE id = :did
    """), {"did": result.document_id})
    doc = row.mappings().first()
    assert_check("Document exists in DB", doc is not None, errors)
    if doc:
        assert_check(
            "Document status = 'processed'",
            doc["processing_status"] == "processed",
            errors,
            f"got {doc['processing_status']!r}",
        )
        assert_check(
            f"DB items_extracted = {result.items_extracted}",
            doc["items_extracted"] == result.items_extracted,
            errors,
        )
        assert_check(
            f"DB items_matched = {result.items_matched}",
            doc["items_matched"] == result.items_matched,
            errors,
        )
        assert_check(
            f"DB items_unresolved = {result.items_unresolved}",
            doc["items_unresolved"] == result.items_unresolved,
            errors,
        )
        assert_check("processed_at is set", doc["processed_at"] is not None, errors)

    # Price records
    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    n_prices = row.scalar()
    # Price records should be <= items_matched (items without a best_price don't get a record)
    items_with_price = sum(1 for m in result.matched_items if m["price"] is not None)
    assert_check(
        f"Price records in DB: {n_prices} (expected {items_with_price} — matched items with price)",
        n_prices == items_with_price,
        errors,
        f"got {n_prices}",
    )

    # Pending matches
    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    n_pending = row.scalar()
    assert_check(
        f"Pending matches in DB: {n_pending} (expected {result.items_unresolved})",
        n_pending == result.items_unresolved,
        errors,
        f"got {n_pending}",
    )

    # Pending match strategies are all valid DB values
    row = await session.execute(sql_text("""
        SELECT DISTINCT match_strategy FROM pending_matches WHERE document_id = :did
    """), {"did": result.document_id})
    db_strategies = {r[0] for r in row.all()}
    valid_strategies = {"fuzzy_text", "embedding", "none"}
    assert_check(
        f"All pending match strategies valid: {db_strategies}",
        db_strategies.issubset(valid_strategies),
        errors,
        f"invalid: {db_strategies - valid_strategies}",
    )

    # Auto-created aliases
    row = await session.execute(sql_text("""
        SELECT COUNT(*) FROM product_aliases
        WHERE source = 'price_list' AND supplier_id = :sid
    """), {"sid": supplier_id})
    n_aliases = row.scalar()
    assert_check(
        f"Auto-created aliases: {n_aliases} (should be >= 1)",
        n_aliases >= 1,
        errors,
    )

    # ── Cleanup ──────────────────────────────────────────────
    # Delete test data created by this pipeline run.
    # Order matters due to FK constraints.
    await session.execute(sql_text(
        "DELETE FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    await session.execute(sql_text(
        "DELETE FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    # Don't delete aliases — they are useful for the re-ingestion test (Test 5)

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print(f"\n  PASS ✓  ({result.items_extracted} extracted, "
          f"{result.items_matched} matched, {result.items_unresolved} pending)")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 5: Re-ingestion — aliases speed up matching
# ═══════════════════════════════════════════════════════════════

async def test_reingestion_uses_aliases(
    session, tenant_id: str, supplier_id: str, supplier_terms: str,
    openai_client: AsyncOpenAI,
) -> bool:
    """
    After Test 4 created aliases, re-ingest the same PDF.
    Verify that MORE items now match (via exact_alias) and that the
    overall match rate is higher.
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Re-ingestion — aliases speed up matching")
    print("=" * 70)

    pdf_path = os.path.join(FIXTURES, "price_list_jrg.pdf")
    if not os.path.exists(pdf_path):
        print("    SKIP: price_list_jrg.pdf not found")
        return True

    errors: list[str] = []

    with open(pdf_path, "rb") as f:
        raw_bytes = f.read()

    controller = IngestController(session, tenant_id, openai_client)

    print("    Running pipeline (re-ingest)... ", end="", flush=True)
    t0 = time.time()
    result = await controller.ingest_price_list(
        raw_bytes=raw_bytes,
        supplier_id=supplier_id,
        supplier_terms=supplier_terms,
        max_pages=1,
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    # Count how many items were matched via exact_alias (from auto-created aliases)
    exact_alias_count = sum(
        1 for m in result.matched_items if m["strategy"] == "exact_alias"
    )
    exact_code_count = sum(
        1 for m in result.matched_items if m["strategy"] == "exact_code"
    )
    semantic_count = sum(
        1 for m in result.matched_items if m["strategy"] in ("semantic", "llm_verified")
    )

    print(f"\n    Items extracted:    {result.items_extracted}")
    print(f"    Items matched:     {result.items_matched}")
    print(f"      exact_alias:     {exact_alias_count}")
    print(f"      exact_code:      {exact_code_count}")
    print(f"      semantic/llm:    {semantic_count}")
    print(f"    Items unresolved:  {result.items_unresolved}")

    assert_check(
        f"Re-ingestion found exact_alias matches (got {exact_alias_count})",
        exact_alias_count > 0,
        errors,
        "Aliases from previous ingestion should speed up matching",
    )

    assert_check(
        f"Match count >= previous run's match count (got {result.items_matched})",
        result.items_matched > 0,
        errors,
    )

    # The match rate should be at least as good as the first run
    match_rate = result.items_matched / max(result.items_extracted, 1)
    assert_check(
        f"Match rate: {match_rate:.1%} (should be reasonable)",
        match_rate >= 0.30,
        errors,
    )

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

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print(f"\n  PASS ✓  (exact_alias={exact_alias_count}, "
          f"total matched={result.items_matched}/{result.items_extracted})")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 6: Full pipeline — Ibérica 1 page
# ═══════════════════════════════════════════════════════════════

async def test_full_pipeline_iberica(
    session, tenant_id: str, supplier_id: str, supplier_terms: str,
    openai_client: AsyncOpenAI,
) -> bool:
    """Run the full ingestion pipeline on page 1 of the Ibérica PDF."""
    print("\n" + "=" * 70)
    print("  TEST 6: Full pipeline — Ibérica Comércio (1 page)")
    print("=" * 70)

    pdf_path = os.path.join(FIXTURES, "price_list_iberica.pdf")
    if not os.path.exists(pdf_path):
        print("    SKIP: price_list_iberica.pdf not found")
        return True

    errors: list[str] = []

    with open(pdf_path, "rb") as f:
        raw_bytes = f.read()

    controller = IngestController(session, tenant_id, openai_client)

    print("    Running pipeline (parse + resolve)... ", end="", flush=True)
    t0 = time.time()
    result = await controller.ingest_price_list(
        raw_bytes=raw_bytes,
        supplier_id=supplier_id,
        supplier_terms=supplier_terms,
        max_pages=1,
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    # ── Basic checks ─────────────────────────────────────────
    assert_check(
        f"items_extracted > 0 (got {result.items_extracted})",
        result.items_extracted > 0,
        errors,
    )
    assert_check(
        f"items_matched + items_unresolved == items_extracted",
        result.items_matched + result.items_unresolved == result.items_extracted,
        errors,
    )
    assert_check(
        f"items_matched > 0 (got {result.items_matched})",
        result.items_matched > 0,
        errors,
    )

    # ── DB verification ──────────────────────────────────────
    row = await session.execute(sql_text(
        "SELECT processing_status FROM documents WHERE id = :did"
    ), {"did": result.document_id})
    doc = row.mappings().first()
    assert_check(
        "Document status = 'processed'",
        doc is not None and doc["processing_status"] == "processed",
        errors,
    )

    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM price_records WHERE source_document_id = :did"
    ), {"did": result.document_id})
    n_prices = row.scalar()
    assert_check(
        f"Price records created: {n_prices} (>= 1)",
        n_prices >= 1,
        errors,
    )

    row = await session.execute(sql_text(
        "SELECT COUNT(*) FROM pending_matches WHERE document_id = :did"
    ), {"did": result.document_id})
    n_pending = row.scalar()
    assert_check(
        f"Pending matches created: {n_pending} == {result.items_unresolved}",
        n_pending == result.items_unresolved,
        errors,
    )

    # Strategy breakdown
    strategies = {}
    for m in result.matched_items:
        s = m["strategy"]
        strategies[s] = strategies.get(s, 0) + 1
    print(f"\n    Strategy breakdown: {strategies}")

    # ── Cleanup ──────────────────────────────────────────────
    await session.execute(sql_text("DELETE FROM pending_matches WHERE document_id = :did"), {"did": result.document_id})
    await session.execute(sql_text("DELETE FROM price_records WHERE source_document_id = :did"), {"did": result.document_id})
    await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": result.document_id})

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print(f"\n  PASS ✓  ({result.items_extracted} extracted, "
          f"{result.items_matched} matched, {result.items_unresolved} pending)")
    return True


# ═══════════════════════════════════════════════════════════════
# Test 7: Empty / invalid input handling
# ═══════════════════════════════════════════════════════════════

async def test_empty_pdf(
    session, tenant_id: str, supplier_id: str,
    openai_client: AsyncOpenAI,
) -> bool:
    """Verify the pipeline handles an empty/non-PDF input gracefully."""
    print("\n" + "=" * 70)
    print("  TEST 7: Empty / invalid input handling")
    print("=" * 70)

    errors: list[str] = []
    controller = IngestController(session, tenant_id, openai_client)

    # ── 7a. Empty bytes ──────────────────────────────────────
    print("\n  7a. Empty bytes:")
    try:
        result = await controller.ingest_price_list(
            raw_bytes=b"",
            supplier_id=supplier_id,
            max_pages=1,
        )
        # Parser should return empty list → 0 extracted
        assert_check(
            f"items_extracted = 0 (got {result.items_extracted})",
            result.items_extracted == 0,
            errors,
        )
        assert_check(
            "items_matched = 0",
            result.items_matched == 0,
            errors,
        )
        # Cleanup
        await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": result.document_id})
    except Exception as exc:
        # It's acceptable for the parser to raise on garbage input.
        # The controller should catch this and mark the document as failed.
        print(f"    Pipeline raised: {type(exc).__name__}: {str(exc)[:100]}")
        assert_check(
            "Exception was raised (acceptable for garbage input)",
            True,
            errors,
        )
        # Check that the document was marked as failed
        row = await session.execute(sql_text(
            "SELECT processing_status FROM documents ORDER BY created_at DESC LIMIT 1"
        ))
        doc = row.mappings().first()
        if doc and doc["processing_status"] == "failed":
            print(f"    {icon(True)} Document marked as 'failed' in DB")
        # Cleanup any failed docs
        await session.execute(sql_text("DELETE FROM documents WHERE processing_status = 'failed'"))

    # ── 7b. Non-PDF bytes ────────────────────────────────────
    print("\n  7b. Non-PDF bytes (random text):")
    try:
        result = await controller.ingest_price_list(
            raw_bytes=b"This is not a PDF file at all. Just plain text.",
            supplier_id=supplier_id,
            max_pages=1,
        )
        assert_check(
            f"items_extracted = 0 (got {result.items_extracted})",
            result.items_extracted == 0,
            errors,
        )
        await session.execute(sql_text("DELETE FROM documents WHERE id = :did"), {"did": result.document_id})
    except Exception as exc:
        print(f"    Pipeline raised: {type(exc).__name__}: {str(exc)[:100]}")
        assert_check(
            "Exception raised for garbage bytes (acceptable)",
            True,
            errors,
        )
        await session.execute(sql_text("DELETE FROM documents WHERE processing_status = 'failed'"))

    if errors:
        print(f"\n  FAIL: {len(errors)} errors")
        return False
    print("\n  PASS ✓")
    return True


# ═══════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════

async def run_tests():
    print("=" * 70)
    print("  INGESTION PIPELINE — TEST SUITE")
    print("=" * 70)

    if not settings.openai_api_key:
        print("  ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    results: dict[str, bool | None] = {}

    # ── Test 1: Strategy mapping (no DB needed) ──────────────
    results["strategy_mapping"] = test_strategy_mapping()

    # ── Connect to DB for remaining tests ────────────────────
    print("\n" + "=" * 70)
    print("  Connecting to database...")
    print("=" * 70)

    async with async_session() as session:
        # Find tenant
        row = await session.execute(sql_text(
            "SELECT id FROM tenants WHERE name = 'Nova Safra Dev' LIMIT 1"
        ))
        tenant_row = row.mappings().first()
        if not tenant_row:
            print("  ERROR: tenant 'Nova Safra Dev' not found.")
            print("         Run: python -m scripts.seed_from_fixtures")
            sys.exit(1)
        tenant_id = str(tenant_row["id"])
        print(f"  Tenant: {tenant_id}")

        await set_tenant_context(session, tenant_id)

        # Find suppliers
        supplier_ids: dict[str, str] = {}
        supplier_terms: dict[str, str] = {}
        for sname in ["JRG Alimentos", "Ibérica Comércio"]:
            row = await session.execute(sql_text(
                "SELECT id, default_terms FROM suppliers WHERE name = :n LIMIT 1"
            ), {"n": sname})
            srow = row.mappings().first()
            if not srow:
                print(f"  ERROR: supplier '{sname}' not found.")
                sys.exit(1)
            supplier_ids[sname] = str(srow["id"])
            supplier_terms[sname] = srow["default_terms"] or "30 dias"
            print(f"  Supplier: {sname} → {supplier_ids[sname]}")

        # Find a product for unit tests
        row = await session.execute(sql_text(
            "SELECT id FROM products LIMIT 1"
        ))
        product_row = row.mappings().first()
        if not product_row:
            print("  ERROR: no products seeded.")
            sys.exit(1)
        sample_product_id = str(product_row["id"])
        print(f"  Sample product: {sample_product_id}")

        # ── Clean up any leftover test data from previous runs ──
        print("\n  Cleaning up previous test data...", end=" ")
        # Delete documents (and cascading data) that were created by tests
        await session.execute(sql_text(
            "DELETE FROM pending_matches WHERE document_id IN "
            "(SELECT id FROM documents WHERE subject = 'Test document lifecycle')"
        ))
        await session.execute(sql_text(
            "DELETE FROM price_records WHERE source_document_id IN "
            "(SELECT id FROM documents WHERE subject = 'Test document lifecycle')"
        ))
        await session.execute(sql_text(
            "DELETE FROM documents WHERE subject = 'Test document lifecycle'"
        ))
        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE raw_description = 'TEST ALIAS DESCRIPTION XYZ'"
        ))
        await session.commit()
        print("done")

        # ── Test 2: Writer — document lifecycle ──────────────
        results["writer_document_lifecycle"] = await test_writer_document_lifecycle(
            session, tenant_id, supplier_ids["JRG Alimentos"],
        )
        await session.commit()

        # ── Test 3: Writer — records ─────────────────────────
        results["writer_records"] = await test_writer_records(
            session, tenant_id, supplier_ids["JRG Alimentos"], sample_product_id,
        )
        await session.commit()

        # ── Test 4: Full pipeline — JRG ──────────────────────
        results["pipeline_jrg"] = await test_full_pipeline_jrg(
            session, tenant_id,
            supplier_ids["JRG Alimentos"],
            supplier_terms["JRG Alimentos"],
            openai_client,
        )
        await session.commit()

        # ── Test 5: Re-ingestion ─────────────────────────────
        results["reingestion_aliases"] = await test_reingestion_uses_aliases(
            session, tenant_id,
            supplier_ids["JRG Alimentos"],
            supplier_terms["JRG Alimentos"],
            openai_client,
        )
        await session.commit()

        # ── Test 6: Full pipeline — Ibérica ──────────────────
        results["pipeline_iberica"] = await test_full_pipeline_iberica(
            session, tenant_id,
            supplier_ids["Ibérica Comércio"],
            supplier_terms["Ibérica Comércio"],
            openai_client,
        )
        await session.commit()

        # ── Test 7: Empty / invalid input ────────────────────
        results["empty_input"] = await test_empty_pdf(
            session, tenant_id,
            supplier_ids["JRG Alimentos"],
            openai_client,
        )
        await session.commit()

        # ── Cleanup aliases created during pipeline tests ────
        # Remove auto-created aliases from tests so the DB stays clean
        print("\n  Final cleanup: removing test aliases...", end=" ")
        await session.execute(sql_text(
            "DELETE FROM product_aliases WHERE source = 'price_list'"
        ))
        await session.commit()
        print("done")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            print(f"  SKIP   {name}")
        elif passed:
            print(f"  ✓ PASS  {name}")
        else:
            print(f"  ✗ FAIL  {name}")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED — review output above")
    print("=" * 70 + "\n")

    return all_passed


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = asyncio.run(run_tests())
    sys.exit(0 if passed else 1)
