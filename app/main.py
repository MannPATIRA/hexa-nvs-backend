from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Query
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.database import get_db
from app.middleware.tenant import get_tenant_session
from app.services.ingestion.controller import IngestController

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


@app.post("/api/v1/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    supplier_id: str = Form(...),
    max_pages: int | None = Query(default=None, description="Limit PDF pages to parse (default: all)"),
    db: AsyncSession = Depends(get_tenant_session),
):
    """
    Upload a supplier price list PDF and run the full ingestion pipeline:
    parse → match → store prices / queue pending matches.
    """
    # Get tenant_id from session context
    result = await db.execute(text("SELECT current_setting('app.current_tenant', true)"))
    tenant_id = result.scalar()

    # Validate the supplier exists and fetch default terms
    sup_result = await db.execute(text("""
        SELECT id, default_terms FROM suppliers WHERE id = :sid
    """), {"sid": supplier_id})
    sup_row = sup_result.mappings().first()
    if not sup_row:
        raise HTTPException(status_code=404, detail="Supplier not found")

    # Read uploaded file
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Run the ingestion pipeline
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    controller = IngestController(db, tenant_id, openai_client)

    result = await controller.ingest_price_list(
        raw_bytes=raw_bytes,
        supplier_id=supplier_id,
        supplier_terms=sup_row["default_terms"],
        max_pages=max_pages,
    )

    return {
        "document_id": result.document_id,
        "items_extracted": result.items_extracted,
        "items_matched": result.items_matched,
        "items_unresolved": result.items_unresolved,
        "matched": result.matched_items,
        "pending_review": result.pending_items,
    }