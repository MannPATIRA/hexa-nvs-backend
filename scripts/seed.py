"""
Run: python -m scripts.seed
Seeds the database with test data for development.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from sqlalchemy import text
from app.database import async_session

async def seed():
    async with async_session() as session:
        # Check if already seeded
        result = await session.execute(text("SELECT COUNT(*) FROM tenants"))
        if result.scalar() > 0:
            print("Database already has data. Run the DROP script first to reseed.")
            return

        tenant_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        # Tenant
        await session.execute(text("""
            INSERT INTO tenants (id, name, currency, language, po_number_prefix, erp_type)
            VALUES (:id, :name, 'BRL', 'pt-BR', 'NS', 'mock')
        """), {"id": str(tenant_id), "name": "Nova Safra Dev"})

        # User
        await session.execute(text("""
            INSERT INTO users (id, tenant_id, email, name, role)
            VALUES (:id, :tid, 'buyer@novasafra.dev', 'Test Buyer', 'admin')
        """), {"id": str(user_id), "tid": str(tenant_id)})

        # Suppliers
        suppliers = {}
        supplier_data = [
            ("Itambé", "fiscal@itambe.dev", "30 dias", ["dairy"]),
            ("Fritz & Frida", "vendas@fritzfrida.dev", "45 dias", ["canned_goods"]),
            ("Moinho Sul", "marcos@moinhosul.dev", "28 dias", ["dry_goods"]),
            ("Antonella", "vendas@antonella.dev", "30 dias", ["dry_goods"]),
            ("Piracanjuba", "vendas@piracanjuba.dev", "30 dias", ["dairy"]),
        ]
        for name, email, terms, cats in supplier_data:
            sid = uuid4()
            await session.execute(text("""
                INSERT INTO suppliers (id, tenant_id, name, contact_emails, default_terms, categories)
                VALUES (:id, :tid, :name, :emails, :terms, :cats)
            """), {
                "id": str(sid), "tid": str(tenant_id), "name": name,
                "emails": [email], "terms": terms, "cats": cats,
            })
            suppliers[name] = sid

        # Products
        products = {}
        product_data = [
            ("Creme de Leite Itambé 1.01kg", "dairy", "caixa", 12),
            ("Farinha de Trigo Antonella 1kg", "dry_goods", "caixa", 10),
            ("Champignon Fatiado Fritz&Frida", "canned_goods", "caixa", 24),
            ("Alcaparras Fritz&Frida", "condiments", "caixa", 12),
            ("Farinha de Milho Antonella 1kg", "dry_goods", "caixa", 10),
            ("Creme de Leite Piracanjuba 1kg", "dairy", "caixa", 12),
            ("Requeijão Itambé 400g", "dairy", "caixa", 6),
            ("Azeitona Verde Fritz&Frida", "canned_goods", "caixa", 12),
        ]
        for cname, cat, unit, pack in product_data:
            pid = uuid4()
            await session.execute(text("""
                INSERT INTO products (id, tenant_id, canonical_name, category, unit, pack_size)
                VALUES (:id, :tid, :name, :cat, :unit, :pack)
            """), {
                "id": str(pid), "tid": str(tenant_id),
                "name": cname, "cat": cat, "unit": unit, "pack": pack,
            })
            products[cname] = pid

        # Aliases (simulate confirmed matches from previous use)
        aliases = [
            ("Creme de Leite Itambé 1.01kg", "Itambé", "CREME DE LEITE UHT 17% GORD ITAMBE 1,01KG"),
            ("Farinha de Trigo Antonella 1kg", "Antonella", "FARINHA DE TRIGO TIPO 1 ANTONELLA 1KG CX10"),
            ("Farinha de Trigo Antonella 1kg", "Moinho Sul", "FRH TRIGO ANTONELLA TP1 1KG CX10"),
            ("Champignon Fatiado Fritz&Frida", "Fritz & Frida", "CHAMPIGNON FATIADO FRITZ FRIDA CX24"),
            ("Creme de Leite Piracanjuba 1kg", "Piracanjuba", "CREME DE LEITE UHT PIRACANJUBA 1KG CX12"),
        ]
        for pname, sname, raw in aliases:
            await session.execute(text("""
                INSERT INTO product_aliases
                    (tenant_id, product_id, supplier_id, raw_description, normalised_description, source, confirmed)
                VALUES (:tid, :pid, :sid, :raw, :norm, 'onboarding', true)
            """), {
                "tid": str(tenant_id), "pid": str(products[pname]),
                "sid": str(suppliers[sname]), "raw": raw, "norm": raw.lower().strip(),
            })

        # Price records (4 weeks of history)
        price_data = [
            ("Creme de Leite Itambé 1.01kg", "Itambé", [54.00, 53.00, 52.50, 52.00]),
            ("Creme de Leite Piracanjuba 1kg", "Piracanjuba", [52.00, 51.50, 51.50, 51.00]),
            ("Farinha de Trigo Antonella 1kg", "Antonella", [36.00, 37.00, 37.50, 38.50]),
            ("Farinha de Trigo Antonella 1kg", "Moinho Sul", [35.50, 36.00, 36.50, 37.00]),
            ("Champignon Fatiado Fritz&Frida", "Fritz & Frida", [28.00, 28.00, 28.50, 28.00]),
        ]
        for pname, sname, prices in price_data:
            for i, price in enumerate(prices):
                obs_date = now - timedelta(weeks=3 - i)
                await session.execute(text("""
                    INSERT INTO price_records
                        (tenant_id, product_id, supplier_id, unit_price, payment_terms, source_type, observed_at)
                    VALUES (:tid, :pid, :sid, :price, :terms, 'price_list', :obs)
                """), {
                    "tid": str(tenant_id), "pid": str(products[pname]),
                    "sid": str(suppliers[sname]), "price": price,
                    "terms": supplier_data[[s[0] for s in supplier_data].index(sname)][2],
                    "obs": obs_date,
                })

        await session.commit()

        print("=" * 50)
        print("SEED COMPLETE")
        print("=" * 50)
        print(f"Tenant ID: {tenant_id}")
        print(f"User ID:   {user_id}")
        print(f"Suppliers: {len(suppliers)}")
        print(f"Products:  {len(products)}")
        print(f"Aliases:   {len(aliases)}")
        print(f"Prices:    {len(price_data) * 4}")
        print()
        print("Use this header for all API calls:")
        print(f"  X-Tenant-Id: {tenant_id}")
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(seed())