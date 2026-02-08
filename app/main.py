from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.tenant import get_tenant_session

app = FastAPI(title="Hexa Backend")

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT COUNT(*) FROM tenants"))
    return {"status": "ok", "tenants": result.scalar()}

@app.get("/api/v1/suppliers")
async def list_suppliers(db: AsyncSession = Depends(get_tenant_session)):
    result = await db.execute(text("""
        SELECT id, name, contact_emails, default_terms, categories, reliability_score
        FROM suppliers
        WHERE tenant_id::text = current_setting('app.current_tenant', true)
        ORDER BY name
    """))
    rows = result.mappings().all()
    return {"suppliers": [dict(r) for r in rows]}

@app.get("/api/v1/products")
async def list_products(db: AsyncSession = Depends(get_tenant_session)):
    result = await db.execute(text("""
        SELECT id, canonical_name, category, unit, pack_size
        FROM products
        WHERE tenant_id::text = current_setting('app.current_tenant', true)
        ORDER BY canonical_name
    """))
    rows = result.mappings().all()
    return {"products": [dict(r) for r in rows]}

@app.get("/api/v1/products/{product_id}/prices")
async def price_history(product_id: str, db: AsyncSession = Depends(get_tenant_session)):
    result = await db.execute(text("""
        SELECT pr.unit_price, pr.payment_terms, pr.source_type, pr.observed_at,
               s.name as supplier_name
        FROM price_records pr
        JOIN suppliers s ON s.id = pr.supplier_id
        WHERE pr.product_id = :pid
          AND pr.tenant_id::text = current_setting('app.current_tenant', true)
          AND s.tenant_id::text = current_setting('app.current_tenant', true)
        ORDER BY pr.observed_at DESC
    """), {"pid": product_id})
    rows = result.mappings().all()
    return {"prices": [dict(r) for r in rows]}