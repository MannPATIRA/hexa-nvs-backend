"""
Comprehensive product resolver integration tests.

Loads ALL extracted items from JSON cache (produced by seed_from_fixtures.py),
resolves every single one, and reports a detailed scoreboard.

Prerequisite:
  1. python -m scripts.migrate_add_embeddings   (one-time pgvector setup)
  2. python -m scripts.seed_from_fixtures        (parse PDFs, seed DB)

Run:
  python tests/test_product_resolver.py
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import async_session, set_tenant_context
from app.services.ingestion.product_resolver import ProductResolver, MatchResult
from scripts.seed_from_fixtures import normalize_to_canonical, accent_insensitive_key, CACHE_FILES

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

PASS_THRESHOLD = 0.90   # 90 % match rate required to pass


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def load_cached_items(supplier_name: str) -> list[dict]:
    """Load extracted items from JSON cache.  Errors if cache is missing."""
    cache_path = CACHE_FILES[supplier_name]
    if not os.path.exists(cache_path):
        print(f"  ERROR: cache file not found: {cache_path}")
        print("         Run: python -m scripts.seed_from_fixtures")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def icon(ok: bool) -> str:
    return "✓" if ok else "✗"


# ═══════════════════════════════════════════════════════════════
# Main test runner
# ═══════════════════════════════════════════════════════════════

async def run_tests():

    # ── Pre-flight ─────────────────────────────────────────
    print("=" * 70)
    print("  COMPREHENSIVE PRODUCT RESOLVER TEST")
    print("=" * 70)

    if not settings.openai_api_key:
        print("  ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # ── Step 1: Load items from cache ──────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Load extracted items from cache")
    print("=" * 70)

    items_by_supplier: dict[str, list[dict]] = {}
    for sname in CACHE_FILES:
        items = load_cached_items(sname)
        items_by_supplier[sname] = items
        print(f"  {sname}: {len(items)} items")

    total_items = sum(len(v) for v in items_by_supplier.values())
    print(f"  Total: {total_items} items")

    # ── Step 2: Build expectation map ──────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Build expectation map")
    print("=" * 70)

    # Map: (supplier_name, raw_description) → expected canonical name
    expectations: dict[tuple[str, str], str] = {}
    canonical_names: set[str] = set()

    for sname, items in items_by_supplier.items():
        for item in items:
            raw = item["raw_description"]
            expected = normalize_to_canonical(raw)
            expectations[(sname, raw)] = expected
            canonical_names.add(expected)

    print(f"  Unique canonical products: {len(canonical_names)}")
    print(f"  Expected matches: {total_items} / {total_items}  (100%)")

    # ── Step 3: Connect to DB ──────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: Connect to database")
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

        # Find supplier IDs
        supplier_ids: dict[str, str] = {}
        for sname in items_by_supplier:
            row = await session.execute(sql_text(
                "SELECT id FROM suppliers WHERE name = :n LIMIT 1"
            ), {"n": sname})
            srow = row.mappings().first()
            if not srow:
                print(f"  ERROR: supplier '{sname}' not found in DB")
                sys.exit(1)
            supplier_ids[sname] = str(srow["id"])
            print(f"  Supplier: {sname} → {supplier_ids[sname]}")

        # Check embeddings
        emb_count = await session.execute(sql_text(
            "SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL"
        ))
        n_emb = emb_count.scalar()
        print(f"  Products with embeddings: {n_emb}")
        if n_emb == 0:
            print("  ERROR: no product embeddings. Run seed first.")
            sys.exit(1)

        resolver = ProductResolver(session, openai_client, use_llm_verification=True)

        # ── Step 4: Exact match quick tests ────────────────
        print("\n" + "=" * 70)
        print("  STEP 4: Exact match quick tests (pre-seeded aliases)")
        print("=" * 70)

        exact_tests_passed = True

        # 4a. Exact code match — JRG code "01674"
        print("\n  4a. Exact supplier-code match (code=01674):")
        code_item = None
        for item in items_by_supplier.get("JRG Alimentos", []):
            if item.get("supplier_code") == "01674":
                code_item = item
                break
        if code_item:
            expected_name = normalize_to_canonical(code_item["raw_description"])
            result = await resolver.resolve(
                raw_description=code_item["raw_description"],
                supplier_id=supplier_ids["JRG Alimentos"],
                supplier_code="01674",
            )
            ok = (
                result.strategy == "exact_code"
                and result.confidence == 1.0
                and accent_insensitive_key(result.product_name or "")
                == accent_insensitive_key(expected_name)
            )
            if not ok:
                exact_tests_passed = False
            print(f"      {icon(ok)} code=01674  desc=\"{code_item['raw_description'][:50]}\"")
            print(f"         strategy={result.strategy}  confidence={result.confidence}")
            print(f"         matched: \"{result.product_name}\"")
            print(f"         expected: \"{expected_name}\"")
        else:
            print("      ✗ Code 01674 not found in JRG cache")
            exact_tests_passed = False

        # 4b. Exact alias text match — Ibérica "AÇÚCAR DE COCO (QUILO)"
        print("\n  4b. Exact alias text match:")
        alias_raw = "AÇÚCAR DE COCO (QUILO)"
        alias_expected = normalize_to_canonical(alias_raw)
        result = await resolver.resolve(
            raw_description=alias_raw,
            supplier_id=supplier_ids["Ibérica Comércio"],
        )
        ok = (
                result.strategy == "exact_alias"
                and result.confidence == 1.0
                and accent_insensitive_key(result.product_name or "")
                == accent_insensitive_key(alias_expected)
            )
        if not ok:
            exact_tests_passed = False
        print(f"      {icon(ok)} desc=\"{alias_raw}\"")
        print(f"         strategy={result.strategy}  confidence={result.confidence}")
        print(f"         matched: \"{result.product_name}\"")
        print(f"         expected: \"{alias_expected}\"")

        # ── Step 5: Resolve ALL items ──────────────────────
        print("\n" + "=" * 70)
        print("  STEP 5: Resolve ALL extracted items")
        print("=" * 70)

        # Tracking
        correct_matches: list[dict] = []
        wrong_matches: list[dict] = []
        no_matches: list[dict] = []
        strategy_counts: dict[str, int] = {}
        strategy_confidences: dict[str, list[float]] = {}

        overall_t0 = time.time()

        for sname, items in items_by_supplier.items():
            sid = supplier_ids[sname]
            print(f"\n  Resolving {len(items)} items from {sname}...")
            t0 = time.time()

            for idx, item in enumerate(items):
                raw = item["raw_description"]
                expected = expectations[(sname, raw)]

                result = await resolver.resolve(
                    raw_description=raw,
                    supplier_id=sid,
                    supplier_code=item.get("supplier_code"),
                )

                # Track strategy counts
                strat = result.strategy
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
                if strat != "none":
                    strategy_confidences.setdefault(strat, []).append(result.confidence)

                # Classify result
                detail = {
                    "supplier": sname,
                    "raw_description": raw,
                    "expected_canonical": expected,
                    "actual_product": result.product_name,
                    "strategy": result.strategy,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                }

                if result.product_name == expected:
                    correct_matches.append(detail)
                elif (
                    result.product_name is not None
                    and accent_insensitive_key(result.product_name)
                    == accent_insensitive_key(expected)
                ):
                    # Same product, accent variant — count as correct
                    correct_matches.append(detail)
                elif result.product_name is not None:
                    wrong_matches.append(detail)
                else:
                    no_matches.append(detail)

                # Progress indicator
                done = idx + 1
                if done % 25 == 0 or done == len(items):
                    print(f"    ... {done}/{len(items)}")

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s "
                  f"({elapsed / max(len(items), 1):.2f}s per item)")

        overall_elapsed = time.time() - overall_t0

        # ── Step 6: Scoreboard ─────────────────────────────
        print("\n" + "=" * 70)
        print("  SCOREBOARD")
        print("=" * 70)

        actual_matched = len(correct_matches) + len(wrong_matches)
        match_rate = actual_matched / total_items if total_items > 0 else 0
        correct_rate = len(correct_matches) / total_items if total_items > 0 else 0

        print(f"\n  Total extracted items:       {total_items}")
        print(f"  Unique canonical products:   {len(canonical_names)}")
        print(f"  Expected matches:            {total_items}  (100%)")
        print(f"  Actual matches:              {actual_matched}  ({match_rate:.1%})")
        print(f"    Correct match:             {len(correct_matches)}")
        print(f"    Wrong match:               {len(wrong_matches)}")
        print(f"    No match (strategy=none):  {len(no_matches)}")
        print(f"  Correct rate:                {correct_rate:.1%}")
        print(f"  Time elapsed:                {overall_elapsed:.1f}s")

        # Strategy breakdown
        print(f"\n  {'Strategy':<16} {'Count':>6} {'Avg Conf':>10}")
        print("  " + "-" * 36)
        for strat in ["exact_code", "exact_alias", "semantic", "llm_verified", "none"]:
            cnt = strategy_counts.get(strat, 0)
            if strat in strategy_confidences and strategy_confidences[strat]:
                avg = sum(strategy_confidences[strat]) / len(strategy_confidences[strat])
                print(f"  {strat:<16} {cnt:>6}     {avg:>6.3f}")
            else:
                print(f"  {strat:<16} {cnt:>6}        {'—':>4}")

        # ── Failures detail ────────────────────────────────
        if wrong_matches:
            print(f"\n  WRONG MATCHES ({len(wrong_matches)}):")
            print("  " + "-" * 66)
            for d in wrong_matches:
                print(f"    [{d['supplier'][:8]}] \"{d['raw_description'][:55]}\"")
                print(f"      expected: \"{d['expected_canonical']}\"")
                print(f"      actual:   \"{d['actual_product']}\"  "
                      f"(strategy={d['strategy']}, conf={d['confidence']:.3f})")
                if d["reasoning"]:
                    print(f"      reason:   {d['reasoning']}")

        if no_matches:
            print(f"\n  NO MATCHES ({len(no_matches)}):")
            print("  " + "-" * 66)
            for d in no_matches[:30]:
                print(f"    [{d['supplier'][:8]}] \"{d['raw_description'][:55]}\"")
                print(f"      expected: \"{d['expected_canonical']}\"")
            if len(no_matches) > 30:
                print(f"    ... and {len(no_matches) - 30} more")

        # ── Step 7: Pass/fail verdict ──────────────────────
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)

        passed_exact = exact_tests_passed
        passed_threshold = correct_rate >= PASS_THRESHOLD

        print(f"  Exact match tests:   {'PASS' if passed_exact else 'FAIL'}")
        print(f"  Match rate:          {correct_rate:.1%}  "
              f"(threshold: {PASS_THRESHOLD:.0%})  "
              f"{'PASS' if passed_threshold else 'FAIL'}")

        all_passed = passed_exact and passed_threshold

        if all_passed:
            print("\n  ALL TESTS PASSED ✓")
        else:
            print("\n  SOME TESTS FAILED — review output above")
        print("=" * 70 + "\n")

        return all_passed


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = asyncio.run(run_tests())
    sys.exit(0 if passed else 1)
