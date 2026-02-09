"""
Run: python -m scripts.seed_from_fixtures

Parses the first 10 pages of each PDF fixture, caches results to JSON,
then seeds the database with a comprehensive canonical product catalog
derived from the extracted items.

On first run: parses PDFs (~5-10 min, costs ~$2-5 in OpenAI calls).
On subsequent runs: loads from JSON cache (~instant).

Prerequisite: run  python -m scripts.migrate_add_embeddings  first.
"""
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from uuid import uuid4

from openai import AsyncOpenAI
from sqlalchemy import text

from app.config import settings
from app.database import async_session
from app.services.ingestion.parsers.pdf_parser import PdfLlmParser
from app.services.ingestion.product_resolver import ProductResolver

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "test_fixtures")
PAGES_TO_PARSE = 10

CACHE_FILES = {
    "Ibérica Comércio": os.path.join(FIXTURES, "cache_iberica_10p.json"),
    "JRG Alimentos":    os.path.join(FIXTURES, "cache_jrg_10p.json"),
}

PDF_FILES = {
    "Ibérica Comércio": os.path.join(FIXTURES, "price_list_iberica.pdf"),
    "JRG Alimentos":    os.path.join(FIXTURES, "price_list_jrg.pdf"),
}

SUPPLIERS = [
    # (name, email, default_terms, categories)
    ("Ibérica Comércio", "vendas@ibericacomercio.com.br", "30 dias",
     ["dried_fruit", "sweetener", "spice", "additive"]),
    ("JRG Alimentos", "comercial@jrgalimentos.com.br", "30 dias",
     ["dried_fruit", "sweetener", "spice", "additive"]),
]

# Brand names to strip from descriptions when normalizing
_BRANDS_TO_STRIP = [
    "La Terre", "Native", "Docican", "Mãe Terra",
    "Leve Crock", "Qualitá", "Yoki",
]


# ═══════════════════════════════════════════════════════════════
# Normalization — converts a supplier description to a canonical
# internal product name.  Used by both seed and test scripts.
# ═══════════════════════════════════════════════════════════════

def accent_insensitive_key(s: str) -> str:
    """
    Return a key with acute, grave, and circumflex accents stripped for
    dedup/comparison.  Preserves cedilla (ç) and tilde (ã, õ) which are
    essential in Portuguese.

    Used by the seed script (accent-insensitive dedup of canonical names)
    and by the test (accent-insensitive comparison of expected vs actual).

    Examples:
      "Açúcar" → "Açucar"      (ú→u, ç preserved)
      "Açafrão" → "Açafrão"    (ã preserved, ç preserved)
      "Ácido" → "Acido"        (Á→A)
      "Amêndoa" → "Amendoa"    (ê→e)
    """
    # Combining marks to strip: grave (0300), acute (0301), circumflex (0302)
    # Keep: tilde (0303), cedilla (0327)
    _STRIP = {"\u0300", "\u0301", "\u0302"}
    decomposed = unicodedata.normalize("NFKD", s)
    cleaned = "".join(c for c in decomposed if c not in _STRIP)
    return unicodedata.normalize("NFC", cleaned)


def normalize_to_canonical(description: str) -> str:
    """
    Convert a raw supplier description into a canonical base product name.

    Strips packaging variants, container/size info, brand names, and
    applies title-case.  Multiple line items (different sizes of the
    same product) will produce the SAME canonical name.

    NOTE: Accents are PRESERVED in the output.  Use accent_insensitive_key()
    for dedup/comparison when two suppliers spell the same word differently
    (e.g. "Açucar" vs "Açúcar").

    Examples:
      "ABACAXI CRISTALIZADO (QUILO)"                → "Abacaxi Cristalizado"
      "ABACAXI CRISTALIZADO (CAIXA)"                → "Abacaxi Cristalizado"
      "ABACAXI DESIDRATADO C/ AÇÚCAR (QUILO)"       → "Abacaxi Desidratado Com Açúcar"
      "ABACAXI EM CALDA VIDRO 400GR (CAIXA)"        → "Abacaxi Em Calda Vidro 400Gr"
      "Alho em flocos desidratado cx 20 kg"          → "Alho Em Flocos Desidratado"
      "Açúcar Demerara Organico La Terre 15x1 kg"    → "Açúcar Demerara Organico"
      "Açúcar Mascavo Docican 20 kg"                 → "Açúcar Mascavo"
      "Cereja desidratada importada 6,25 kg"         → "Cereja Desidratada Importada"
      "Açucar Demerara Organico Native 5 kg (Novo)"  → "Açucar Demerara Organico"
      "ARROZ JASMINE (QUILO)"                        → "Arroz Jasmine"
    """
    name = description.strip()

    # 0. Strip "(Novo)" marker — JRG uses this for new items
    name = re.sub(r"\s*\(Novo\)\s*$", "", name, flags=re.IGNORECASE)

    # 1. Strip Ibérica-style packaging variants at end
    name = re.sub(
        r"\s*\((CAIXA|MEIA-CAIXA|QUILO|UNIDADE)\)\s*$",
        "", name, flags=re.IGNORECASE,
    )

    # 2. Expand abbreviations
    name = re.sub(r"\bC/", "Com ", name)

    # 3. Strip JRG-style container + size  (e.g. "cx 20 kg", "sc 5 kg", "cx2 kg",
    #    "cx 4x1 kg", "cx 11,34 kg").  Handles Brazilian comma-decimal numbers.
    name = re.sub(
        r"\b(cx|sc|fd|pct)\s*\d+(?:[,.]\d+)?(\s*[xX]\s*\d+(?:[,.]\d+)?)?\s*(kg|g|un|l|ml)?\b",
        "", name, flags=re.IGNORECASE,
    )

    # 4. Strip known brand names
    for brand in _BRANDS_TO_STRIP:
        name = re.sub(
            r"\b" + re.escape(brand) + r"\b",
            "", name, flags=re.IGNORECASE,
        )

    # 5a. Strip trailing size patterns WITH space before unit
    #     Matches: " 20 kg", " 15x1 kg", " 6 X 5 kg", " 2,5 kg", " 6,25 kg"
    #     Does NOT match attached sizes like "400GR" (those are product attributes).
    name = re.sub(
        r"\s+\d+(?:[,.]\d+)?(\s*[xX]\s*\d+(?:[,.]\d+)?)?\s+(kg|g|gr|un|l|ml)\s*$",
        "", name, flags=re.IGNORECASE,
    )

    # 5b. Strip trailing attached sizes ONLY for kg (which are pack sizes).
    #     "2KG" is a pack size, but "400GR", "500G", "200ML" are product attributes.
    name = re.sub(
        r"\s+\d+(?:[,.]\d+)?kg\s*$",
        "", name, flags=re.IGNORECASE,
    )

    # 6. Strip trailing plain numbers (pack sizes after brand removal): " 20", " 5", " 2,5"
    name = re.sub(r"\s+\d+(?:[,.]\d+)?(\s*[xX]\s*\d+(?:[,.]\d+)?)?\s*$", "", name)

    # 7. Clean up and title-case
    name = re.sub(r"\s+", " ", name).strip()
    name = name.title()

    return name


# ═══════════════════════════════════════════════════════════════
# PDF parsing + JSON caching
# ═══════════════════════════════════════════════════════════════

def _parse_and_cache(supplier_name: str) -> list[dict]:
    """Parse a PDF and save results to a JSON cache file."""
    pdf_path = PDF_FILES[supplier_name]
    cache_path = CACHE_FILES[supplier_name]

    if not os.path.exists(pdf_path):
        print(f"  ERROR: {pdf_path} not found")
        sys.exit(1)

    print(f"  Parsing {supplier_name} ({PAGES_TO_PARSE} pages)...")
    t0 = time.time()
    parser = PdfLlmParser(max_pages=PAGES_TO_PARSE)
    with open(pdf_path, "rb") as f:
        items = parser.parse(f.read())
    elapsed = time.time() - t0
    print(f"    Extracted {len(items)} items in {elapsed:.1f}s")

    # Serialize to JSON-friendly dicts
    serialized = []
    for item in items:
        serialized.append({
            "raw_description": item.raw_description,
            "supplier_code": item.supplier_code,
            "price_cash": item.price_cash,
            "price_credit": item.price_credit,
            "currency": item.currency,
            "unit": item.unit,
            "pack_size": item.pack_size,
            "category": item.category,
            "is_promotional": item.is_promotional,
        })

    cache_data = {
        "supplier": supplier_name,
        "pages_parsed": PAGES_TO_PARSE,
        "item_count": len(serialized),
        "items": serialized,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"    Cached to {os.path.basename(cache_path)}")

    return serialized


def load_or_parse(supplier_name: str, force_parse: bool = False) -> list[dict]:
    """Load items from cache, or parse the PDF if cache is missing."""
    cache_path = CACHE_FILES[supplier_name]

    if not force_parse and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {supplier_name}: loaded {data['item_count']} items from cache "
              f"({data['pages_parsed']} pages)")
        return data["items"]

    return _parse_and_cache(supplier_name)


# ═══════════════════════════════════════════════════════════════
# Seed logic
# ═══════════════════════════════════════════════════════════════

async def seed():
    force = "--force-parse" in sys.argv

    # ── 1. Load / parse items ──────────────────────────────
    print("=" * 70)
    print("  STEP 1: Load extracted items")
    print("=" * 70)

    items_by_supplier: dict[str, list[dict]] = {}
    for sname in CACHE_FILES:
        items_by_supplier[sname] = load_or_parse(sname, force_parse=force)

    # ── 2. Build canonical product catalog ─────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Build canonical product catalog")
    print("=" * 70)

    raw_canonical_names: set[str] = set()

    total_items = 0
    for sname, items in items_by_supplier.items():
        for item in items:
            raw = item["raw_description"]
            canonical = normalize_to_canonical(raw)
            raw_canonical_names.add(canonical)
            total_items += 1

    # Accent-insensitive dedup: merge canonicals that differ only in accents.
    # Among duplicates, keep the version with the most diacritics (most correct
    # Portuguese spelling).  E.g. "Açúcar De Coco" wins over "Açucar De Coco".
    accent_groups: dict[str, str] = {}   # accent_key → preferred canonical
    for name in raw_canonical_names:
        key = accent_insensitive_key(name)
        if key not in accent_groups:
            accent_groups[key] = name
        else:
            # Keep the version with more combining marks (more accented)
            existing = accent_groups[key]
            if len(unicodedata.normalize("NFKD", name)) > len(unicodedata.normalize("NFKD", existing)):
                accent_groups[key] = name

    canonical_names = set(accent_groups.values())
    sorted_names = sorted(canonical_names)

    duped = len(raw_canonical_names) - len(canonical_names)
    if duped:
        print(f"  Accent-insensitive dedup: merged {duped} duplicate(s)")
    print(f"  Total extracted items:      {total_items}")
    print(f"  Unique canonical products:  {len(sorted_names)}")
    for name in sorted_names:
        print(f"    - {name}")

    # Helper: map a raw description to the deduped canonical name
    def to_deduped_canonical(desc: str) -> str:
        raw = normalize_to_canonical(desc)
        key = accent_insensitive_key(raw)
        return accent_groups.get(key, raw)

    # ── 3. Determine aliases ───────────────────────────────
    # Alias 1: exact CODE match — JRG code "01674"
    alias1_canonical = None
    for item in items_by_supplier.get("JRG Alimentos", []):
        if item.get("supplier_code") == "01674":
            alias1_canonical = to_deduped_canonical(item["raw_description"])
            alias1_raw = item["raw_description"]
            break

    # Alias 2: exact TEXT match — Ibérica "AÇÚCAR DE COCO (QUILO)"
    alias2_raw = "AÇÚCAR DE COCO (QUILO)"
    alias2_canonical = to_deduped_canonical(alias2_raw)

    aliases_to_create = []
    if alias1_canonical and alias1_canonical in canonical_names:
        aliases_to_create.append(
            (alias1_canonical, "JRG Alimentos", alias1_raw, "01674")
        )
    if alias2_canonical in canonical_names:
        aliases_to_create.append(
            (alias2_canonical, "Ibérica Comércio", alias2_raw, None)
        )

    # ── 4. Seed database ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: Seed database")
    print("=" * 70)

    async with async_session() as session:
        # Drop existing data
        print("  Dropping existing data...")
        for table in [
            "po_line_items", "purchase_orders", "procurement_queue",
            "pending_matches", "documents", "audit_log",
            "price_records", "product_aliases", "products",
            "suppliers", "users", "tenants",
        ]:
            await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
        await session.commit()

        # Tenant + user
        tenant_id = uuid4()
        user_id = uuid4()
        await session.execute(text("""
            INSERT INTO tenants (id, name, currency, language, po_number_prefix, erp_type)
            VALUES (:id, :name, 'BRL', 'pt-BR', 'NS', 'mock')
        """), {"id": str(tenant_id), "name": "Nova Safra Dev"})
        await session.execute(text("""
            INSERT INTO users (id, tenant_id, email, name, role)
            VALUES (:id, :tid, 'buyer@novasafra.dev', 'Test Buyer', 'admin')
        """), {"id": str(user_id), "tid": str(tenant_id)})

        # Suppliers
        supplier_ids: dict[str, str] = {}
        for name, email, terms, cats in SUPPLIERS:
            sid = uuid4()
            await session.execute(text("""
                INSERT INTO suppliers (id, tenant_id, name, contact_emails, default_terms, categories)
                VALUES (:id, :tid, :name, :emails, :terms, :cats)
            """), {
                "id": str(sid), "tid": str(tenant_id), "name": name,
                "emails": [email], "terms": terms, "cats": cats,
            })
            supplier_ids[name] = str(sid)

        # Products (one per unique canonical name)
        product_ids: dict[str, str] = {}
        for cname in sorted_names:
            pid = uuid4()
            await session.execute(text("""
                INSERT INTO products (id, tenant_id, canonical_name, category, unit)
                VALUES (:id, :tid, :name, :cat, :unit)
            """), {
                "id": str(pid), "tid": str(tenant_id),
                "name": cname, "cat": "general", "unit": "kg",
            })
            product_ids[cname] = str(pid)
        print(f"  Inserted {len(product_ids)} canonical products")

        # Aliases
        for pname, sname, raw, code in aliases_to_create:
            await session.execute(text("""
                INSERT INTO product_aliases
                    (tenant_id, product_id, supplier_id, raw_description,
                     normalised_description, supplier_code, source, confirmed)
                VALUES (:tid, :pid, :sid, :raw, :norm, :code, 'onboarding', true)
            """), {
                "tid": str(tenant_id),
                "pid": product_ids[pname],
                "sid": supplier_ids[sname],
                "raw": raw,
                "norm": raw.lower().strip(),
                "code": code,
            })
        print(f"  Inserted {len(aliases_to_create)} aliases")
        await session.commit()

        # Generate embeddings
        if not settings.openai_api_key:
            print("\n  WARNING: No OPENAI_API_KEY — skipping embeddings.")
        else:
            print("  Generating embeddings...")
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resolver = ProductResolver(session, client, use_llm_verification=False)
            count = await resolver.ensure_embeddings()
            print(f"  Generated {count} embeddings.")

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SEED FROM FIXTURES — COMPLETE")
    print("=" * 70)
    print(f"  Tenant ID:              {tenant_id}")
    print(f"  Suppliers:              {len(supplier_ids)}")
    print(f"  Canonical products:     {len(product_ids)}")
    print(f"  Total extracted items:  {total_items}")
    print(f"  Aliases:                {len(aliases_to_create)}")
    for pname, sname, raw, code in aliases_to_create:
        tag = f"code={code}" if code else f'text="{raw[:40]}"'
        print(f"    {pname} <- {sname} ({tag})")
    print()
    print(f"  X-Tenant-Id: {tenant_id}")
    print()
    print("  Next: python tests/test_product_resolver.py")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(seed())
