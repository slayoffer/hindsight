"""
Example of using memory-poc as a library in another project.

This demonstrates how to:
1. Import and use the memory system directly
2. Import and extend the FastAPI app
3. Mount the memory app as a sub-application
"""
import asyncio
from web import app, memory


async def example_memory_usage():
    """Example of using the memory system directly."""
    # Initialize memory system
    await memory.initialize()

    try:
        # Store some memories
        await memory.put_async(
            agent_id="example_agent",
            content="Example memory content",
            context="test context"
        )

        # Search for memories
        results, trace = await memory.search_async(
            agent_id="example_agent",
            query="example",
            top_k=5
        )

        print(f"Found {len(results)} results")
        for result in results:
            print(f"  - {result['text']} (score: {result['score']:.4f})")

        # Use think functionality
        think_result = await memory.think_async(
            agent_id="example_agent",
            query="What do you know?",
            thinking_budget=50
        )

        print(f"\nThink result: {think_result['text']}")
        if think_result.get('new_opinions'):
            print(f"New opinions formed: {len(think_result['new_opinions'])}")

    finally:
        # Clean up
        await memory.close()


def example_fastapi_extension():
    """Example of extending the FastAPI app with custom endpoints."""
    from fastapi import FastAPI

    # Option 1: Add endpoints directly to the imported app
    @app.get("/api/custom")
    async def custom_endpoint():
        return {"message": "Custom endpoint added to memory-poc app"}

    # Option 2: Mount as sub-application
    main_app = FastAPI(title="My Application")

    @main_app.get("/")
    async def root():
        return {"message": "My main application"}

    # Mount the memory app at /memory
    main_app.mount("/memory", app)

    # Now you can access:
    # - / -> your main app
    # - /memory/ -> memory visualization
    # - /memory/api/graph -> memory graph API
    # - /memory/api/search -> memory search API

    return main_app


if __name__ == "__main__":
    # Example 1: Use memory system directly
    print("=" * 80)
    print("Example 1: Direct memory system usage")
    print("=" * 80)
    asyncio.run(example_memory_usage())

    # Example 2: FastAPI extension
    print("\n" + "=" * 80)
    print("Example 2: FastAPI app extension")
    print("=" * 80)
    extended_app = example_fastapi_extension()
    print("FastAPI app extended successfully")
    print("To run: uvicorn library_usage_example:extended_app --reload")
