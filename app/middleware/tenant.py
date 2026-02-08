from fastapi import Request, HTTPException
from app.database import set_tenant_context, reset_session, async_session
from sqlalchemy import text

async def get_tenant_session(request: Request):
    """
    FastAPI dependency that:
    1. Reads tenant_id from header (dev mode)
    2. Creates a DB session with RLS scoped to that tenant
    """
    tenant_id = request.headers.get("X-Tenant-Id")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="X-Tenant-Id header required")

    async with async_session() as session:
        try:
            await set_tenant_context(session, tenant_id)
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await reset_session(session)