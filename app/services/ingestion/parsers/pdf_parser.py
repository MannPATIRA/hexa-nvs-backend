"""
PDF price list parser using LLM with structured output.

Uses OpenAI function calling to force the model to return data matching
a Pydantic schema exactly. No JSON parsing hacks needed — the API
guarantees the output matches the schema.

Flow:
  1. Extract raw text from PDF with pdfplumber
  2. Call OpenAI with a tool whose parameters schema is our Pydantic model
  3. The model is forced to call the tool with valid structured data
  4. Parse tool call arguments directly into Pydantic objects
"""

import json
import re
import pdfplumber
import io
from openai import OpenAI
from pydantic import BaseModel, Field
from .base import ParseStrategy, ExtractedLineItem
from app.config import settings


# ═══════════════════════════════════════════════════════════════
# Pydantic models for structured output
# ═══════════════════════════════════════════════════════════════

class ExtractedItem(BaseModel):
    description: str = Field(
        description="Full product name exactly as written in the price list. Include the complete name: base product, brand, and variant/pack (e.g. 'Açucar Demerara Organico La Terre 15x1 kg (Novo)', 'Açucar Mascavo Docican 20 kg'). Do not truncate, abbreviate, or drop any part."
    )
    supplier_code: str | None = Field(default=None, description="Product code/código if present, otherwise null")
    price_cash: float | None = Field(default=None, description="Cash/à vista price as a number (e.g. 43.27). null if R$0,00 or not listed")
    price_credit: float | None = Field(default=None, description="Credit/à prazo price as a number. null if R$0,00 or not listed")
    pack_size: float | None = Field(default=None, description="Weight or quantity number (e.g. 5 from '5 Kg'). null if not listed")
    unit: str | None = Field(default=None, description="Unit of measurement lowercase: kg, un, cx, sc, g, l, ml. null if not listed")
    category: str | None = Field(default=None, description="Product category if listed in the price list, otherwise null")


class PriceListExtraction(BaseModel):
    items: list[ExtractedItem] = Field(description="Every product line item extracted from the price list")
    supplier_name: str | None = Field(default=None, description="Name of the supplier if visible in the document header")
    document_date: str | None = Field(default=None, description="Date of the price list if visible, in YYYY-MM-DD format")


# ═══════════════════════════════════════════════════════════════
# Tool definition for OpenAI API (function calling)
# ═══════════════════════════════════════════════════════════════

def _openai_tool_schema() -> dict:
    schema = PriceListExtraction.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": "save_extracted_price_list",
            "description": "Save the structured data extracted from a supplier price list.",
            "parameters": schema,
        },
    }


EXTRACTION_TOOL = _openai_tool_schema()

SYSTEM_PROMPT = """You are a data extraction system for Brazilian supplier price lists.
Your job is to extract every product line item from the provided text and call the save_extracted_price_list function with the structured data.

Rules:
- Prices are in BRL. Convert "R$ 43,27" to 43.27 (comma = decimal, period = thousands separator)
- If a price is R$ 0,00 or 0,00, set it to null — this means the item is unavailable
- Extract EVERY product row. Do not skip any.
- Do not invent data. Only extract what is explicitly in the text.
- If there are two price columns (à vista and à prazo), extract both.
- If there is only one price column, put it in price_cash.
- For the product description: copy the ENTIRE product name exactly as written. Include every part: base name, brand, and variant/pack (e.g. "La Terre 15x1", "Native 5 kg", "Native 6 X 5", "Docican 20", "Docican 5"). Do not truncate, shorten, or drop the end of the name."""


# ═══════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════

class PdfLlmParser(ParseStrategy):

    def __init__(self, max_pages: int | None = 1, timeout: float = 90.0):
        """
        max_pages: Maximum number of pages to extract from the PDF (default 1).
                   Set to None to process all pages.
        timeout: Seconds to wait for the OpenAI API (default 90). Prevents hanging.
        """
        self.client = OpenAI(
            api_key=settings.openai_api_key or None,
            timeout=timeout,
        )
        self.max_pages = max_pages

    @staticmethod
    def _parse_price(s: str) -> float | None:
        """Parse Brazilian price string (e.g. 'R$ 43,27') to float. Returns None for empty or R$ 0,00."""
        if not s or not s.strip():
            return None
        s = s.strip().replace("R$", "").strip()
        if not s or s in ("0,00", "0"):
            return None
        s = s.replace(".", "").replace(",", ".")
        try:
            v = float(s)
            return v if v != 0 else None
        except ValueError:
            return None

    @staticmethod
    def _parse_pack_size(s: str) -> tuple[float | None, str | None]:
        """Parse pack size string (e.g. '5 Kg', '15 Un') to (number, unit). Unit normalized to lowercase."""
        if not s or not s.strip():
            return (None, None)
        s = s.strip()
        # Match number (optional comma decimal) and optional unit (letters)
        m = re.match(r"^([\d,]+)\s*([a-zA-Z]+)?$", s)
        if not m:
            return (None, None)
        num_str, unit = m.group(1), m.group(2)
        try:
            num = float(num_str.replace(",", "."))
        except ValueError:
            return (None, None)
        unit = unit.lower() if unit else None
        if unit and len(unit) <= 4:
            unit = {"kg": "kg", "un": "un", "unid": "un", "cx": "cx", "sc": "sc", "g": "g", "l": "l", "ml": "ml"}.get(unit, unit)
        return (num, unit)

    def parse(self, raw_bytes: bytes, metadata: dict | None = None) -> list[ExtractedLineItem]:
        # 1. Extract raw text from PDF
        text = self._extract_text(raw_bytes)
        if not text or len(text.strip()) < 50:
            return []

        # 2. Call LLM in chunks (keep chunks small so the API responds in reasonable time)
        all_items = []
        chunks = self._split_into_chunks(text, max_chars=6000)

        for chunk in chunks:
            extraction = self._call_llm(chunk)
            if extraction:
                for item in extraction.items:
                    line = self._to_line_item(item)
                    if line:
                        all_items.append(line)

        return all_items

    def _extract_text(self, raw_bytes: bytes) -> str:
        pages_text = []
        pdf = pdfplumber.open(io.BytesIO(raw_bytes))
        pages = pdf.pages[: self.max_pages] if self.max_pages is not None else pdf.pages
        for page in pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        pdf.close()
        return "\n\n".join(pages_text)

    def _split_into_chunks(self, text: str, max_chars: int = 12000) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        lines = text.split("\n")
        header = "\n".join(lines[:5])
        chunks = []
        current = header + "\n"

        for line in lines[5:]:
            if len(current) + len(line) > max_chars:
                chunks.append(current)
                current = header + "\n"
            current += line + "\n"

        if current.strip() != header.strip():
            chunks.append(current)

        return chunks if chunks else [text]

    def _call_llm(self, text: str) -> PriceListExtraction | None:
        """
        Call OpenAI with function calling to get structured output.
        The model is forced to call our function, so the response
        matches the PriceListExtraction schema.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=8000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract all items from this price list:\n\n{text}"},
            ],
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "save_extracted_price_list"}},
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return None
        for tc in message.tool_calls:
            if tc.function.name == "save_extracted_price_list" and tc.function.arguments:
                return PriceListExtraction.model_validate(json.loads(tc.function.arguments))
        return None

    @staticmethod
    def _to_line_item(item: ExtractedItem) -> ExtractedLineItem | None:
        """Convert a Pydantic ExtractedItem to our internal ExtractedLineItem."""
        if len(item.description.strip()) < 3:
            return None

        price_cash = item.price_cash if item.price_cash and item.price_cash > 0 else None
        price_credit = item.price_credit if item.price_credit and item.price_credit > 0 else None

        # Skip items where both prices are null/zero
        if not price_cash and not price_credit:
            return None

        return ExtractedLineItem(
            raw_description=item.description.strip(),
            supplier_code=item.supplier_code,
            price_cash=price_cash,
            price_credit=price_credit,
            unit=item.unit,
            pack_size=item.pack_size if item.pack_size and item.pack_size > 0 else None,
            category=item.category,
            is_promotional="promoção" in (item.category or "").lower()
                          or "promoção" in item.description.lower(),
        )