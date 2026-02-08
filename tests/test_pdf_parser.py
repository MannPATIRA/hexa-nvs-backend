"""
Parser tests against real Brazilian supplier price lists.

Usage:
  1. Place the real PDFs in test_fixtures/:
       test_fixtures/price_list_iberica.pdf
       test_fixtures/price_list_jrg.pdf

  2. Run:
       python tests/test_pdf_parser.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ingestion.parsers.pdf_parser import PdfLlmParser

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "test_fixtures")


# ═══════════════════════════════════════════════════════════════
# Known values read directly from the real PDFs (via screenshots)
# ═══════════════════════════════════════════════════════════════

IBERICA_SPOT_CHECKS = [
    # (substring in description, expected_cash, expected_credit)
    ("ABACAXI CRISTALIZADO (CAIXA)", 43.27, 45.00),
    ("ABACAXI CRISTALIZADO (MEIA-CAIXA)", 56.94, 59.22),
    ("ABACAXI CRISTALIZADO (QUILO)", 62.25, 64.74),
    ("ABACAXI DESIDRATADO C/ AÇÚCAR (CAIXA)", 74.10, 77.06),
    ("ABACAXI DESIDRATADO C/ AÇÚCAR (MEIA-CAIXA)", 81.65, 84.92),
    ("ABACAXI DESIDRATADO C/ AÇÚCAR (QUILO)", 87.94, 91.46),
    ("ABACAXI EM CALDA VIDRO 400GR (CAIXA)", 27.78, 28.89),
    ("ABACAXI EM CALDA VIDRO 400GR (UNIDADE)", 34.78, 36.17),
    ("ABACAXI GLACEADO (MEIA-CAIXA)", 57.28, 59.57),
    ("ABACAXI GLACEADO (QUILO)", 62.63, 65.14),
    ("AÇAFRÃO (CÚRCUMA) (CAIXA)", 7.23, 7.52),
    ("AÇAFRÃO (CÚRCUMA) (MEIA-CAIXA)", 8.40, 8.74),
    ("AÇAFRÃO (CÚRCUMA) (QUILO)", 9.45, 9.83),
    ("AÇAFRÃO (CÚRCUMA) PREMIUM (CAIXA)", 9.04, 9.40),
    ("AÇAFRÃO (CÚRCUMA) PREMIUM (MEIA-CAIXA)", 10.49, 10.91),
    ("AÇAFRÃO (CÚRCUMA) PREMIUM (QUILO)", 11.81, 12.28),
    ("ÁCIDO CÍTRICO (CAIXA)", 16.01, 16.65),
    ("ÁCIDO CÍTRICO (MEIA-CAIXA)", 18.36, 19.09),
    ("ÁCIDO CÍTRICO (QUILO)", 20.48, 21.30),
    ("AÇÚCAR DE COCO (CAIXA)", 20.40, 21.22),
    ("AÇÚCAR DE COCO (MEIA-CAIXA)", 23.40, 24.34),
    ("AÇÚCAR DE COCO (QUILO)", 26.10, 27.14),
    ("AÇÚCAR DE MAÇÃ (CAIXA)", 61.79, 64.26),
    ("AÇÚCAR DE MAÇÃ (MEIA-CAIXA)", 68.30, 71.03),
    ("AÇÚCAR DE MAÇÃ (QUILO)", 73.87, 76.82),
    ("AÇÚCAR DEMERARA (CAIXA)", 5.43, 5.65),
]

JRG_SPOT_CHECKS = [
    # (code, description_contains, expected_cash, expected_credit, expected_pack_size)
    ("01674", "Abacaxi em rodela cristalizado", 54.90, 56.60, 3.0),
    ("21843", "Abacaxi em rodela desidratado", 49.47, 51.00, 5.0),
    ("01538", "desidratado em pó", 86.33, 89.00, 2.0),
    ("20294", "coco", 20.86, 21.50, 10.0),
    ("20125", "coco", 22.60, 23.30, 5.0),
    ("21454", "coco", 23.67, 24.40, 2.0),
    ("01090", "maçã", 52.87, 54.50, 5.0),
    ("02387", "La Terre 15x1", 5.72, 5.90, 15.0),
    ("02388", "La Terre 6x5", 5.43, 5.60, 30.0),
    ("00064", "Native 12x1", 6.60, 6.80, 12.0),
    ("01032", "Native 5 kg", 7.95, 8.20, 5.0),
    ("02823", "Native 6 X 5", 7.57, 7.80, 30.0),
    ("00065", "Mascavo Docican 20", 6.31, 6.50, 20.0),
    ("01189", "Mascavo Docican 5", 7.66, 7.90, 5.0),
    ("00235", "Agar Agar", 66.45, 68.50, 5.0),
    ("01543", "Agar Agar", 73.82, 76.10, 2.0),
    ("02161", "Alcachofra em folhas", 41.71, 43.00, 1.0),
    ("01340", "Alcachofra em folhas", 36.67, 37.80, 5.0),
    ("20100", "Alecrim flocos", 7.18, 7.40, 25.0),
    ("20350", "Alecrim flocos", 8.63, 8.90, 5.0),
    ("21455", "Alecrim flocos", 9.99, 10.30, 2.0),
    ("00160", "Alfarroba em pó", 88.17, 90.90, 4.0),
    ("01580", "Alfarroba em pó", 97.78, 100.80, 2.0),
    ("10114", "Alho em flocos desidratado", 27.16, 28.00, 20.0),
    ("20923", "Alho em flocos desidratado", 29.39, 30.30, 5.0),
    ("21393", "Alho em flocos desidratado", 32.01, 33.00, 2.0),
]

# These codes have R$ 0,00 in both columns — parser should skip them
JRG_ZERO_PRICE_CODES = ["01810", "00279", "00434", "00541"]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def find_item(items, description_substring):
    """Find item by description substring. Handles near-exact and fuzzy."""
    sub = description_substring.lower()
    # Exact substring match first
    matches = [i for i in items if sub in i.raw_description.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Pick the one whose description starts closest to the substring
        matches.sort(key=lambda i: len(i.raw_description))
        return matches[0]
    return None


def find_by_code(items, code):
    matches = [i for i in items if i.supplier_code == code]
    return matches[0] if matches else None


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_price_parsing():
    print("=" * 70)
    print("TEST: Price string parsing")
    print("=" * 70)

    parse = PdfLlmParser._parse_price
    cases = [
        ("R$ 43,27", 43.27),
        ("R$ 1.234,56", 1234.56),
        ("R$43,27", 43.27),
        ("43,27", 43.27),
        ("R$ 0,00", None),
        ("0,00", None),
        ("", None),
        ("R$ 5,43", 5.43),
        ("R$ 100,80", 100.80),
        ("R$ 86,33", 86.33),
        ("R$ 7,52", 7.52),
    ]
    errors = []
    for val, expected in cases:
        result = parse(val)
        if result != expected:
            errors.append(f"parse_price({val!r}) = {result}, expected {expected}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print(f"  ✓ All {len(cases)} cases passed")
    return True


def test_pack_size_parsing():
    print("\n" + "=" * 70)
    print("TEST: Pack size parsing")
    print("=" * 70)

    parse = PdfLlmParser._parse_pack_size
    cases = [
        ("5 Kg", (5.0, "kg")),
        ("15 Un", (15.0, "un")),
        ("1 Kg", (1.0, "kg")),
        ("25 Kg", (25.0, "kg")),
        ("10 Kg", (10.0, "kg")),
        ("3 Kg", (3.0, "kg")),
        ("1 Un", (1.0, "un")),
        ("3", (3.0, None)),
        ("2,5", (2.5, None)),
        ("5", (5.0, None)),
        ("30", (30.0, None)),
        ("12", (12.0, None)),
        ("", (None, None)),
    ]
    errors = []
    for val, expected in cases:
        result = parse(val)
        if result != expected:
            errors.append(f"parse_pack_size({val!r}) = {result}, expected {expected}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print(f"  ✓ All {len(cases)} cases passed")
    return True


def test_iberica():
    path = os.path.join(FIXTURES, "price_list_iberica.pdf")
    if not os.path.exists(path):
        print(f"\n  SKIP: {path} not found")
        return None

    print("\n" + "=" * 70)
    print("TEST: Ibérica Comércio")
    print("=" * 70)

    parser = PdfLlmParser()
    with open(path, "rb") as f:
        items = parser.parse(f.read())

    print(f"\nExtracted {len(items)} items\n")
    errors = []

    # ── Minimum count (the visible portion alone has 26 rows) ──
    if len(items) < 26:
        errors.append(f"Only {len(items)} items, expected at least 26 (visible in screenshot)")

    # ── Every item must have at least one price ──
    no_price = [i for i in items if i.best_price is None]
    if no_price:
        errors.append(f"{len(no_price)} items have no price at all")

    # ── Credit >= cash for every item ──
    for item in items:
        if item.price_cash and item.price_credit and item.price_credit < item.price_cash:
            errors.append(f"Credit < cash: '{item.raw_description[:40]}' {item.price_credit} < {item.price_cash}")

    # ── Spot check every visible row ──
    print("Spot checks (26 visible rows):")
    for desc_sub, exp_cash, exp_credit in IBERICA_SPOT_CHECKS:
        item = find_item(items, desc_sub)
        if not item:
            errors.append(f"NOT FOUND: '{desc_sub}'")
            print(f"  ✗ NOT FOUND: '{desc_sub}'")
            continue

        problems = []
        if item.price_cash != exp_cash:
            problems.append(f"cash={item.price_cash} expected {exp_cash}")
        if item.price_credit != exp_credit:
            problems.append(f"credit={item.price_credit} expected {exp_credit}")

        if problems:
            errors.append(f"'{desc_sub}': {', '.join(problems)}")
            print(f"  ✗ {desc_sub}: {', '.join(problems)}")
        else:
            print(f"  ✓ {desc_sub}: R${item.price_cash} / R${item.price_credit}")

    # ── Dump all items for manual inspection ──
    print(f"\n{'Description':<55} {'Pack':>8} {'Cash':>10} {'Credit':>10} {'Cat':<15}")
    print("-" * 108)
    for item in items:
        pack = f"{item.pack_size or ''}{item.unit or ''}"
        cash = f"R${item.price_cash:.2f}" if item.price_cash else "—"
        credit = f"R${item.price_credit:.2f}" if item.price_credit else "—"
        print(f"{item.raw_description[:55]:<55} {pack:>8} {cash:>10} {credit:>10} {(item.category or ''):<15}")

    print()
    if errors:
        print(f"IBÉRICA: {len(errors)} ERRORS")
        for e in errors:
            print(f"  • {e}")
        return False
    print("IBÉRICA: ALL CHECKS PASSED ✓")
    return True


def test_jrg():
    path = os.path.join(FIXTURES, "price_list_jrg.pdf")
    if not os.path.exists(path):
        print(f"\n  SKIP: {path} not found")
        return None

    print("\n" + "=" * 70)
    print("TEST: JRG Alimentos")
    print("=" * 70)

    parser = PdfLlmParser()
    with open(path, "rb") as f:
        items = parser.parse(f.read())

    print(f"\nExtracted {len(items)} items\n")
    errors = []

    # ── Every item must have a supplier code ──
    no_code = [i for i in items if not i.supplier_code]
    if no_code:
        errors.append(f"{len(no_code)} items missing supplier code")
        for i in no_code[:5]:
            errors.append(f"  No code: '{i.raw_description[:50]}'")

    # ── Zero-price items must be filtered out ──
    print("Zero-price filtering:")
    for code in JRG_ZERO_PRICE_CODES:
        found = find_by_code(items, code)
        if found:
            errors.append(f"Code {code} has R$0,00 but was NOT filtered out")
            print(f"  ✗ Code {code} should be excluded (R$0,00)")
        else:
            print(f"  ✓ Code {code} correctly excluded")

    # ── Every remaining item must have a price ──
    no_price = [i for i in items if i.best_price is None]
    if no_price:
        errors.append(f"{len(no_price)} items have no price")

    # ── Credit >= cash ──
    for item in items:
        if item.price_cash and item.price_credit and item.price_credit < item.price_cash:
            errors.append(f"Credit < cash: code {item.supplier_code} {item.price_credit} < {item.price_cash}")

    # ── Spot check every visible row by code ──
    print("\nSpot checks (26 visible rows):")
    for code, desc_sub, exp_cash, exp_credit, exp_pack in JRG_SPOT_CHECKS:
        item = find_by_code(items, code)
        if not item:
            errors.append(f"NOT FOUND: code {code} ({desc_sub})")
            print(f"  ✗ NOT FOUND: code {code}")
            continue

        problems = []
        if item.price_cash != exp_cash:
            problems.append(f"cash={item.price_cash} expected {exp_cash}")
        if item.price_credit != exp_credit:
            problems.append(f"credit={item.price_credit} expected {exp_credit}")
        if exp_pack and item.pack_size != exp_pack:
            problems.append(f"pack={item.pack_size} expected {exp_pack}")
        if desc_sub.lower() not in item.raw_description.lower():
            problems.append(f"desc '{item.raw_description[:30]}' missing '{desc_sub}'")

        if problems:
            errors.append(f"Code {code}: {', '.join(problems)}")
            print(f"  ✗ {code}: {', '.join(problems)}")
        else:
            print(f"  ✓ {code} | {item.raw_description[:45]:<45} R${item.price_cash} / R${item.price_credit}  pack={item.pack_size}")

    # ── Dump all items ──
    print(f"\n{'Code':<8} {'Description':<50} {'Pack':>6} {'Cash':>10} {'Credit':>10}")
    print("-" * 95)
    for item in items:
        code = item.supplier_code or "—"
        pack = str(item.pack_size) if item.pack_size else ""
        cash = f"R${item.price_cash:.2f}" if item.price_cash else "—"
        credit = f"R${item.price_credit:.2f}" if item.price_credit else "—"
        print(f"{code:<8} {item.raw_description[:50]:<50} {pack:>6} {cash:>10} {credit:>10}")

    print()
    if errors:
        print(f"JRG: {len(errors)} ERRORS")
        for e in errors:
            print(f"  • {e}")
        return False
    print("JRG: ALL CHECKS PASSED ✓")
    return True


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HEXA PDF PARSER — TEST SUITE")
    print("=" * 70 + "\n")

    results = {}
    results["price_parsing"] = test_price_parsing()
    results["pack_size"] = test_pack_size_parsing()
    results["iberica"] = test_iberica()
    results["jrg"] = test_jrg()

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
        print("  SOME TESTS FAILED — fix parser and re-run")
    print("=" * 70 + "\n")
    sys.exit(0 if all_passed else 1)