from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"ssl": "require"}
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with async_session() as session:
        yield session

async def set_tenant_context(session: AsyncSession, tenant_id: str):
    # Validate UUID format to prevent SQL injection
    from uuid import UUID
    UUID(tenant_id)  # Raises ValueError if invalid

    # Set tenant context for RLS policies
    # Note: SET ROLE is not used as it's not compatible with Supabase pooler connections
    await session.execute(text(f"SET app.current_tenant = '{tenant_id}'"))

async def reset_session(session: AsyncSession):
    # Reset tenant context (RESET ROLE not needed as we don't set it)
    try:
        await session.execute(text("RESET app.current_tenant"))
    except Exception:
        # Ignore if setting doesn't exist
        pass