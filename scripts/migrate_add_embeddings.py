"""
Run: python -m scripts.migrate_add_embeddings

Enables pgvector and adds embedding columns to products / product_aliases.
Optionally generates embeddings for all existing rows via OpenAI.

Safe to run multiple times — each step is idempotent.
"""
import asyncio
from openai import AsyncOpenAI
from sqlalchemy import text
from app.config import settings
from app.database import async_session
from app.services.ingestion.product_resolver import ProductResolver


async def migrate():
    async with async_session() as session:
        # ── 1. Enable pgvector extension ────────────────────
        print("Enabling pgvector extension...")
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()
        print("  Done.")

        # ── 2. Add embedding columns (idempotent) ──────────
        print("Adding embedding columns...")

        for table in ("products", "product_aliases"):
            # Check if column already exists
            check = await session.execute(text("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = :tbl AND column_name = 'embedding'
            """), {"tbl": table})
            if check.scalar() is None:
                await session.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN embedding vector(512)")
                )
                print(f"  Added embedding column to {table}")
            else:
                print(f"  {table}.embedding already exists — skipping")

        await session.commit()

        # ── 3. Generate embeddings for existing rows ────────
        if not settings.openai_api_key:
            print("\nNo OPENAI_API_KEY set — skipping embedding generation.")
            print("Run this script again after setting the key to backfill embeddings.")
            return

        print("Generating embeddings for rows that don't have one yet...")
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resolver = ProductResolver(session, client, use_llm_verification=False)

        count = await resolver.ensure_embeddings()
        print(f"  Generated {count} new embeddings.")

    print("\nMigration complete.")


if __name__ == "__main__":
    asyncio.run(migrate())
