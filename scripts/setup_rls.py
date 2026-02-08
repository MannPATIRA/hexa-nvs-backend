"""
Run: python -m scripts.setup_rls
Sets up Row Level Security policies for multi-tenant isolation.
"""
import asyncio
from sqlalchemy import text
from app.database import async_session

async def setup_rls():
    async with async_session() as session:
        # List of tables that need tenant isolation
        tables = [
            "tenants",
            "users",
            "suppliers",
            "products",
            "product_aliases",
            "price_records"
        ]

        print("Setting up Row Level Security...")
        print("=" * 50)

        for table in tables:
            print(f"Configuring RLS for {table}...")

            # Enable RLS on the table
            await session.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))

            # Drop existing policies if they exist
            try:
                await session.execute(text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}"))
            except Exception:
                pass

            # Create policy based on app.current_tenant session variable
            # Special handling for tenants table
            if table == "tenants":
                policy_sql = f"""
                    CREATE POLICY tenant_isolation ON {table}
                    FOR ALL
                    USING (id::text = current_setting('app.current_tenant', true))
                """
            else:
                policy_sql = f"""
                    CREATE POLICY tenant_isolation ON {table}
                    FOR ALL
                    USING (tenant_id::text = current_setting('app.current_tenant', true))
                """

            await session.execute(text(policy_sql))
            print(f"  ✓ Enabled RLS and created tenant_isolation policy")

        await session.commit()

        print("=" * 50)
        print("RLS setup complete!")
        print()
        print("Note: RLS is now enforced. Make sure to:")
        print("1. Always provide X-Tenant-Id header in API requests")
        print("2. Use correct tenant UUIDs that exist in the database")

if __name__ == "__main__":
    asyncio.run(setup_rls())
