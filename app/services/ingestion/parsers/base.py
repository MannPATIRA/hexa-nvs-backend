from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ExtractedLineItem:
    raw_description: str
    supplier_code: str | None = None
    price_cash: float | None = None       # Preço à vista
    price_credit: float | None = None     # Preço à prazo
    currency: str = "BRL"
    unit: str | None = None               # KG, UN, CX, etc.
    pack_size: float | None = None        # Weight or quantity per unit
    category: str | None = None           # If the list includes categories
    is_promotional: bool = False

    @property
    def best_price(self) -> float | None:
        """Return the cash price if available, otherwise credit price."""
        if self.price_cash and self.price_cash > 0:
            return self.price_cash
        if self.price_credit and self.price_credit > 0:
            return self.price_credit
        return None


class ParseStrategy(ABC):
    @abstractmethod
    def parse(self, raw_bytes: bytes, metadata: dict | None = None) -> list[ExtractedLineItem]:
        """Parse raw file bytes into structured line items."""
        pass