"""
Pytest configuration and shared fixtures.
"""
import pytest
import pytest_asyncio
import os
import asyncio
from dotenv import load_dotenv
from memory import TemporalSemanticMemory
import asyncpg

load_dotenv()


@pytest_asyncio.fixture(scope="session")
async def memory():
    """
    Provide a shared memory system instance for all tests in the session.
    This avoids reloading the embedding model for every test (saves 3+ seconds per test).
    """
    mem = TemporalSemanticMemory()
    yield mem
    # Ensure cleanup happens at end of session
    try:
        await mem.close()
    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")


@pytest_asyncio.fixture(scope="function")
async def clean_agent(memory):
    """
    Provide a clean agent ID and clean up data after test.
    Uses agent_id='test' for all tests (multi-tenant isolation).
    """
    agent_id = "test"

    # Clean up before test
    await memory.delete_agent(agent_id)

    yield agent_id

    # Clean up after test
    try:
        await memory.delete_agent(agent_id)
    except Exception as e:
        print(f"Warning: Error during agent cleanup: {e}")


@pytest_asyncio.fixture
async def db_connection():
    """
    Provide a database connection for direct DB queries in tests.
    """
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'), statement_cache_size=0)
    yield conn
    try:
        await conn.close()
    except Exception as e:
        print(f"Warning: Error closing connection: {e}")
