"""
Run: python -m scripts.drop
Drops all data from the database (useful for reseeding).
"""
import asyncio
from sqlalchemy import text
from app.database import async_session

async def drop_all():
    async with async_session() as session:
        print("Dropping all data...")
        print("=" * 50)

        # Drop in reverse order to respect foreign key constraints
        tables = [
            "price_records",
            "product_aliases",
            "products",
            "suppliers",
            "users",
            "tenants"
        ]

        for table in tables:
            print(f"Truncating {table}...")
            await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

        await session.commit()

        print("=" * 50)
        print("All data dropped successfully!")
        print()
        print("You can now run: python -m scripts.seed")

if __name__ == "__main__":
    asyncio.run(drop_all())
