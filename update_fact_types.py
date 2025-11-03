#!/usr/bin/env python3
"""Update existing memory_units to have fact_type='world' if NULL."""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))

    # Check if fact_type column exists
    result = await conn.fetchrow("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'memory_units' AND column_name = 'fact_type'
    """)

    if not result:
        print("fact_type column does not exist! Run migrations first:")
        print("  uv run alembic upgrade head")
        await conn.close()
        return

    print("fact_type column exists ✓")

    # Update existing rows
    count = await conn.fetchval("""
        UPDATE memory_units
        SET fact_type = 'world'
        WHERE fact_type IS NULL
        RETURNING (SELECT COUNT(*) FROM memory_units WHERE fact_type IS NULL)
    """)

    print(f"Updated {count} rows with fact_type='world'")

    # Show distribution
    distribution = await conn.fetch("""
        SELECT fact_type, COUNT(*) as count
        FROM memory_units
        GROUP BY fact_type
        ORDER BY count DESC
    """)

    print("\nFact type distribution:")
    for row in distribution:
        print(f"  {row['fact_type']}: {row['count']}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
