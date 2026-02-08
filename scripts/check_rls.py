"""
Run: python -m scripts.check_rls
Checks RLS configuration and user privileges.
"""
import asyncio
from sqlalchemy import text
from app.database import async_session

async def check_rls():
    async with async_session() as session:
        print("Checking RLS Configuration...")
        print("=" * 50)

        # Check current user and privileges
        result = await session.execute(text("SELECT current_user, current_database()"))
        user, db = result.one()
        print(f"Current user: {user}")
        print(f"Current database: {db}")
        print()

        # Check if user has BYPASSRLS privilege
        result = await session.execute(text("""
            SELECT rolname, rolsuper, rolbypassrls
            FROM pg_roles
            WHERE rolname = current_user
        """))
        row = result.one()
        print(f"User: {row[0]}")
        print(f"Is superuser: {row[1]}")
        print(f"Bypass RLS: {row[2]}")
        print()

        if row[2]:
            print("⚠️  WARNING: Current user has BYPASSRLS privilege!")
            print("   RLS policies will NOT be enforced for this user.")
            print()
            print("   Solutions:")
            print("   1. Use a different database user without BYPASSRLS")
            print("   2. Remove BYPASSRLS from current user (may break other things)")
            print("   3. Use Supabase's built-in RLS with service role key")
        else:
            print("✓ User does not have BYPASSRLS - RLS should be enforced")
            print()

        # Check RLS status on tables
        print("RLS Status on Tables:")
        print("-" * 50)
        result = await session.execute(text("""
            SELECT schemaname, tablename, rowsecurity
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('tenants', 'users', 'suppliers', 'products', 'product_aliases', 'price_records')
            ORDER BY tablename
        """))

        for row in result:
            status = "✓ ENABLED" if row[2] else "✗ DISABLED"
            print(f"  {row[1]}: {status}")

        print()
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(check_rls())
