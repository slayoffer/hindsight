# Entity-Aware Memory System for AI Agents

A proof-of-concept memory system that enables AI agents to store, retrieve, and connect memories using temporal, semantic, and entity-based relationships.

## Overview

This system implements a sophisticated graph-based memory architecture where memories are connected through three complementary networks:

1. **Temporal Network** - Memories linked by time proximity
2. **Semantic Network** - Memories linked by meaning similarity
3. **Entity Network** - Memories linked by shared entities (people, organizations, places)

The combination of these three networks enables powerful memory retrieval that goes beyond simple vector search, allowing agents to find relevant memories through multiple pathways.

## Architecture

### Triple Network Design

The system maintains three separate but interconnected memory networks:

1. **World Network** (`fact_type='world'`)
   - General knowledge and facts about the world
   - Information not specific to the agent's actions
   - Example: "Alice works at Google", "Yosemite is in California"

2. **Agent Network** (`fact_type='agent'`)
   - Facts about what the AI agent specifically did
   - Agent's own actions and experiences
   - Example: "The agent helped debug a Python script", "The agent recommended Yosemite"

3. **Opinion Network** (`fact_type='opinion'`)
   - Agent's formed opinions and perspectives
   - Automatically extracted during think operations
   - Includes reasons and confidence scores (0.0-1.0)
   - Immutable once formed (event_date = when opinion was formed)
   - Example: "Python is better for data science than JavaScript (Reasons: has better libraries like pandas and numpy) [confidence: 0.85]"

All three networks share the same infrastructure (temporal/semantic/entity links) but can be searched independently or together. The **think** operation combines all three networks to formulate consistent, contextual answers while forming new opinions.

### Core Concepts

**Memory Units**: Individual sentence-level memories that are:
- Self-contained (pronouns resolved to actual referents)
- Validated to have subject + verb (complete thoughts)
- Embedded as vectors for semantic similarity
- Timestamped for temporal relationships
- Linked to extracted entities
- Classified as either 'world' or 'agent' fact type

**Entity Resolution**: Named entities (PERSON, ORG, GPE, etc.) are:
- Extracted using spaCy NER
- Disambiguated using a scoring algorithm
- Tracked with canonical IDs across all memories
- Used to create strong connections between related memories

### Three Types of Memory Links

#### 1. Temporal Links (Time-Based)
**Purpose**: Connect memories that occurred close together in time

**How it works**:
- When storing a new memory, find all memories within a time window (default: 24 hours)
- Create weighted links based on temporal proximity
- Weight formula: `weight = max(0.3, 1.0 - (time_diff / window_size))`
- Closer in time = stronger link

**Visualization**: Cyan, dashed lines

**Use case**: "What happened recently?" or understanding sequences of events

#### 2. Semantic Links (Meaning-Based)
**Purpose**: Connect memories with similar content/meaning

**How it works**:
- Generate embeddings using local `bge-small-en-v1.5` model (384 dimensions)
- Store embeddings in PostgreSQL with pgvector extension
- When storing a new memory, find top-k similar memories using cosine similarity
- Create links only if similarity exceeds threshold (default: 0.7)
- Weight = cosine similarity score

**Visualization**: Pink, solid lines

**Technology**:
- **SentenceTransformers** - Local embedding model (BAAI/bge-small-en-v1.5)
- **pgvector** - PostgreSQL extension for vector operations
- **HNSW index** - Fast approximate nearest neighbor search

**Use case**: "Tell me about hiking" retrieves all semantically related outdoor activities

#### 3. Entity Links (Identity-Based)
**Purpose**: Connect ALL memories about the same person, organization, or place

**How it works**:
- Extract entities from text using spaCy NER
- Resolve entity identity using disambiguation algorithm:
  - Name similarity (50% weight) - using SequenceMatcher
  - Co-occurring entities (30% weight) - entities that appear together
  - Temporal proximity (20% weight) - recent mentions more likely same entity
- If score > threshold (0.4 for PERSON with exact match, 0.6 otherwise): reuse existing entity
- If score < threshold: create new entity
- Link all memories mentioning the same entity with weight 1.0 (no decay)

**Visualization**: Gold, thick lines

**Technology**:
- **spaCy** (`en_core_web_sm`) - Named Entity Recognition
- **difflib.SequenceMatcher** - String similarity matching

**Use case**: "What does Alice do?" returns ALL memories about Alice (hiking, work at Google, Python project) even if semantically distant

**Critical advantage**: Solves the problem where "Alice loves hiking" wouldn't normally connect to "Alice works at Google" through semantic similarity alone.

### Spreading Activation Search

The search algorithm explores the memory graph using spreading activation:

1. **Entry Points**: Find top-3 semantically similar memories to the query (vector search, similarity ≥ 0.5)
2. **Activation Spreading**: Start with activation = actual similarity score (0.5 to 1.0) at entry points
3. **Graph Traversal**: Follow links to neighbors, spreading activation with decay (0.8 factor)
4. **Thinking Budget**: Limit exploration to N units (controls computational cost)
5. **Dynamic Weighting**: Combine activation, semantic similarity, recency, and frequency:
   ```
   final_weight = w_a × activation + w_s × semantic_similarity + w_r × recency + w_f × frequency

   # Default weights (configurable via search parameters):
   w_a = 0.30  # Activation weight
   w_s = 0.30  # Semantic similarity weight
   w_r = 0.25  # Recency weight
   w_f = 0.15  # Frequency weight

   semantic_similarity = cosine_similarity(query_embedding, memory_embedding)
   recency = 1 / (1 + log(1 + days_since/365))  # Logarithmic decay with 1-year half-life
   frequency = normalized to [0, 1] from log(access_count + 1) / log(10)
   ```

   **Weight Tuning**: All weights are configurable via `search_async()` parameters, enabling benchmark experiments with different scoring strategies (e.g., emphasizing graph structure vs semantic similarity).

   Recency uses logarithmic decay to provide meaningful differentiation over years:
     - Today: 1.000 (100% weight)
       - 1 week: 0.981 (barely any decay)
       - 1 month: 0.927 (still very recent)
       - 3 months: 0.819 (recent)
       - 6 months: 0.714
       - 1 year: 0.591 (half-life point)
       - 2 years: 0.477 ✓
       - 5 years: 0.358 ✓ (clearly different from 2 years!)
       - 10 years: 0.294 ✓

   This ensures old memories (2yr vs 5yr) have different weights, unlike exponential decay.
6. **Return Top-K**: Sort by final weight and return top results

This approach ensures:
- Semantic relevance to query is always considered (default 30% weight)
- Graph structure influences results through activation (default 30% weight)
- Recently accessed memories get boosted (default 25% weight - recency bias)
- Frequently accessed memories get boosted (default 15% weight - importance signal)

### Search Tracing & Debugging

The system includes comprehensive search tracing to understand and debug the search process:

**Enable tracing**:
```python
results, trace = memory.search(
    agent_id="agent_1",
    query="Who works at Google?",
    enable_trace=True  # Returns detailed SearchTrace object
)
```

**Trace captures**:
- Every node visited with parent/child relationships
- All links explored (followed or pruned) with reasons
- Weight calculations broken down by component
- Entry points selected and their similarity scores
- Pruning decisions (already visited, activation too low, budget exhausted)
- Performance metrics for each search phase

**Export trace for visualization**:
```python
# Save trace as JSON for external visualization tools
trace_json = trace.to_json()
with open("trace.json", "w") as f:
    f.write(trace_json)
```

**Use cases**:
- Understanding why certain memories were/weren't retrieved
- Debugging search behavior
- Analyzing link type effectiveness
- Performance profiling
- Building custom visualization layers

See `SEARCH_TRACE.md` for complete trace API documentation and `examples/trace_example.py` for a working demo.

### Self-Contained Memory Units

Every memory unit is self-contained through LLM fact extraction:

**Problem**: "She joined Google last year" - unclear who "she" is

**Solution**: LLM-based fact extraction that:
- Resolves pronouns to actual referents during extraction
- Makes facts readable without original context
- Includes all relevant details (WHO, WHAT, WHERE, WHEN, WHY, HOW)
- Processes facts in parallel for speed

**Result**: "Alice joined Google last year" - fully self-contained

**Technology**:
- LLM fact extraction with detailed prompts for pronoun resolution
- Structured output using Pydantic models
- Batch processing for efficiency

### LLM-Based Fact Extraction

Raw content is processed through an LLM to extract meaningful facts before storage:

**Problem**: Raw text contains noise (greetings, filler words, reactions) that waste storage and reduce retrieval quality

**Solution**: LLM-based extraction with optimized prompting:
- Filters out social pleasantries and non-informative content
- Extracts only facts with substance (biographical, events, opinions, recommendations, descriptions, relationships)
- Creates self-contained statements with subject+action+context
- Categorizes and attributes facts to speakers

**Technology**:
- **OpenAI-compatible API** - Supports Groq (default), OpenAI, and other providers
- **Structured output** - Uses Pydantic models for reliable fact extraction
- **Optimized prompting** - Concise prompts (~300 chars) emphasize dense output with no fluff
- **Automatic chunking** - Large documents (>120k chars) split at sentence boundaries
- **Fast sentence splitting** - Regex-based splitter (no heavy NLP models)
- **Progress tracking** - Logs chunk processing for transparency

**For large documents (e.g., podcast transcripts)**:
- Documents <120k chars: processed in one pass
- Documents >120k chars: automatically chunked at sentence boundaries
- Each chunk kept under ~30k tokens to avoid output token limits
- Facts aggregated across all chunks

### Technology Stack

**Database**:
- PostgreSQL 15+ with extensions:
  - `pgvector` - Vector similarity operations
  - `uuid-ossp` - UUID generation

**Python Libraries**:
- `psycopg2-binary` - PostgreSQL client
- `sentence-transformers` - Local embedding model (bge-small-en-v1.5)
- `torch` - Deep learning framework (for embeddings)
- `spacy` - NLP (NER, dependency parsing, tokenization)
- `langchain-text-splitters` - Intelligent text chunking
- `networkx` - Graph operations
- `pyvis` - Interactive HTML graph visualization
- `rich` - Terminal UI

**Models**:
- BAAI/bge-small-en-v1.5 - Local embedding model (384 dimensions)

## Quick Start

### Prerequisites

1. PostgreSQL 15+ with pgvector extension
2. Python 3.11+

### Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install spaCy model:
   ```bash
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

3. Create database and run schema:
   ```bash
   psql -U postgres -c "CREATE DATABASE memory_poc"
   psql -U postgres -d memory_poc -f schema.sql
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your DATABASE_URL
   ```

### Run Tests

Run the full test suite:
```bash
uv run pytest tests/ -v
```

Run specific test files:
```bash
uv run pytest tests/test_memory_operations.py -v
uv run pytest tests/test_entity_linking.py -v
```

Run a single test:
```bash
uv run pytest tests/test_memory_operations.py::test_put_creates_memory_units -v
```

### Run Demo

```bash
uv run python demos/demo_entity.py
```

This will:
1. Clear previous demo data
2. Store sample memories about Alice, Bob, Google, Yosemite
3. Search for "What does Alice do?"
4. Show entity resolution results
5. Generate interactive HTML graph visualization

Open `memory_graph_interactive.html` in your browser to explore the memory graph!

## Using as a Library (Local Import)

You can import this project from another Poetry project using a local path dependency:

### 1. Add to your project's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
memory-poc = {path = "../memory-poc", develop = true}
```

Or using poetry CLI:
```bash
poetry add ../memory-poc --editable
```

### 2. Import the memory system:

```python
from memory import TemporalSemanticMemory

# Initialize memory
memory = TemporalSemanticMemory()
await memory.initialize()

# Use the memory system
await memory.put_batch_async(
    agent_id="my_agent",
    contents=["Alice works at Google", "Bob loves hiking"],
    event_date=datetime.now(timezone.utc)
)

results, trace = await memory.search_async(
    agent_id="my_agent",
    query="Who works at Google?"
)
```

### 3. Import the FastAPI app:

```python
from web import app, memory

# Use the FastAPI app in your own project
# You can mount it as a sub-application or run it directly
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Example: Extending the FastAPI app

```python
from fastapi import FastAPI
from web import app as memory_app, memory

# Create your own app
my_app = FastAPI()

# Mount the memory app as a sub-application
my_app.mount("/memory", memory_app)

# Add your own endpoints that use the memory system
@my_app.post("/my-custom-endpoint")
async def my_endpoint():
    # Use the shared memory instance
    results, _ = await memory.search_async(
        agent_id="my_agent",
        query="some query"
    )
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(my_app, host="0.0.0.0", port=8000)
```

### Available Exports

**From `memory` package:**
- `TemporalSemanticMemory` - Main memory system class
- `SearchTrace`, `SearchTracer` - Search tracing utilities
- `QueryInfo`, `EntryPoint`, `NodeVisit`, etc. - Trace data structures

**From `web` package:**
- `app` - FastAPI application instance
- `memory` - Shared TemporalSemanticMemory instance

### Web Server

To run the web interface:

```bash
# Development mode with auto-reload
uvicorn web.server:app --reload --port 8000

# Production mode
uvicorn web.server:app --host 0.0.0.0 --port 8000 --workers 4
```

Then open http://localhost:8000 in your browser to access the visualization interface.

## Project Structure

```
memory-poc/
├── memory/                          # Core memory system package
│   ├── temporal_semantic_memory.py  # Main memory system class
│   ├── operations/                  # Modular operation mixins
│   │   ├── embedding_operations.py  # Embedding generation with process pool
│   │   ├── link_operations.py       # Entity, temporal, semantic links
│   │   ├── batch_operations.py      # Placeholder for future extraction
│   │   └── search_operations.py     # Placeholder for future extraction
│   ├── entity_resolver.py           # Entity extraction and disambiguation
│   ├── llm_client.py                # LLM-based fact extraction
│   └── utils.py                     # Utility functions
│
├── demos/                           # Demo scripts
│   └── demo_entity.py              # Main entity-aware demo
│
├── visualizations/                  # Visualization tools
│   └── interactive_graph.py        # Interactive HTML graph (pyvis)
│
├── schema.sql                       # Database schema
├── pyproject.toml                  # Dependencies
└── README.md                       # This file
```

The memory system uses a **mixin pattern** for code organization:
- `TemporalSemanticMemory` inherits from `EmbeddingOperationsMixin` and `LinkOperationsMixin`
- This reduced the main file from 1,720 lines to 1,420 lines (17% reduction)
- See `memory/operations/README.md` for detailed refactoring documentation

## Key Features

✅ **Triple network architecture**: Separate networks for world knowledge, agent actions, and opinions
✅ **Opinion formation**: Automatically extracts and stores opinions with confidence scores during thinking
✅ **Three-layered linking**: Temporal + Semantic + Entity
✅ **Entity disambiguation**: Resolves "Alice" across different contexts
✅ **Self-contained units**: Pronouns resolved to actual referents
✅ **Spreading activation**: Graph-aware search beyond vector similarity
✅ **Think operation**: Combines all three networks for consistent, contextual answers
✅ **Confidence scores**: Opinions include confidence ratings (0.0-1.0) based on supporting evidence
✅ **Interactive visualization**: Explore memory graph in browser
✅ **Recency & frequency weighting**: Recent and important memories boosted
✅ **Linguistic validation**: Memory units verified to have subject + verb
✅ **Modular architecture**: Mixin pattern with 17% code size reduction

## API Usage

### Store Memories

```python
from memory import TemporalSemanticMemory

memory = TemporalSemanticMemory()

memory.put(
    agent_id="agent_1",
    content="Alice works at Google as a software engineer. She joined last year.",
    context="Career discussion",
    event_date=datetime.now(timezone.utc)
)
```

### Search Memories

```python
# Basic search (trace disabled by default)
results, trace = memory.search(
    agent_id="agent_1",
    query="What does Alice do?",
    thinking_budget=50,  # How many units to explore
    top_k=10             # Number of results to return
)

for result in results:
    print(f"{result['text']} (weight: {result['weight']:.3f})")

# Search only world facts
results, trace = memory.search(
    agent_id="agent_1",
    query="What does Alice do?",
    fact_type="world"  # Only search world network
)

# Search only agent facts
results, trace = memory.search(
    agent_id="agent_1",
    query="What have I done?",
    fact_type="agent"  # Only search agent network
)

# Search only opinions
results, trace = memory.search(
    agent_id="agent_1",
    query="What do I think about Python?",
    fact_type="opinion"  # Only search opinion network
)

# Search with tracing for debugging
results, trace = memory.search(
    agent_id="agent_1",
    query="What does Alice do?",
    thinking_budget=50,
    top_k=10,
    enable_trace=True  # Returns detailed SearchTrace object
)

# Analyze trace
print(f"Nodes visited: {trace.summary.total_nodes_visited}")
print(f"Entry points: {len(trace.entry_points)}")
trace_json = trace.to_json()  # Export for visualization

# Search with custom weight tuning
results, trace = memory.search(
    agent_id="agent_1",
    query="What does Alice do?",
    thinking_budget=50,
    top_k=10,
    weight_activation=0.40,   # Emphasize graph structure
    weight_semantic=0.40,     # Emphasize semantic similarity
    weight_recency=0.10,      # De-emphasize recency
    weight_frequency=0.10     # De-emphasize frequency
)
```

### Think and Formulate Answers

The `think` operation combines all three networks to formulate consistent, contextual answers:

```python
result = await memory.think_async(
    agent_id="agent_1",
    query="What do you think about Python?",
    thinking_budget=50,
    top_k=10
)

print(result["text"])  # Plain text answer from LLM

# Access facts used to formulate the answer
for fact in result["based_on"]["world"]:
    print(f"World: {fact['text']}")

for fact in result["based_on"]["agent"]:
    print(f"Agent: {fact['text']}")

for fact in result["based_on"]["opinion"]:
    print(f"Opinion: {fact['text']} (confidence: {fact.get('confidence_score', 'N/A')})")

# Check for newly formed opinions
for opinion in result["new_opinions"]:
    print(f"New opinion formed: {opinion['text']} (confidence: {opinion['confidence']})")
```

The think operation:
1. Searches the agent network to understand the agent's identity and actions
2. Searches the world network for relevant general knowledge
3. Searches the opinion network for existing perspectives
4. Uses an LLM (Groq by default) to formulate a coherent answer, being consistent with existing opinions
5. Extracts any new opinions formed during thinking with confidence scores
6. Stores new opinions with the current timestamp and query context
7. Returns plain text response with supporting facts from all networks and new opinions

## How It Works: Example

**Input memories**:
1. "Alice loves hiking in the mountains" (7 days ago)
2. "She goes hiking every weekend in Yosemite" (7 days ago)
3. "Alice works at Google as a software engineer" (3 days ago)
4. "She joined Google last year" (3 days ago)

**Processing**:
1. ✅ Coreference resolution → "Alice goes hiking...", "Alice joined Google..."
2. ✅ Entity extraction → Identifies "Alice" (PERSON), "Google" (ORG), "Yosemite" (GPE)
3. ✅ Entity resolution → All "Alice" mentions = same person
4. ✅ Create links:
   - Temporal: Memory 1 ↔ Memory 2 (same day)
   - Semantic: "hiking" memories link together, "Google" memories link together
   - Entity: ALL Alice memories strongly linked (weight 1.0)

**Query: "What does Alice do?"**
1. Vector search finds "Alice works at Google" as top entry point
2. Spreading activation follows entity links to find:
   - "Alice joined Google..." (entity link: Alice)
   - "Alice loves hiking..." (entity link: Alice)
   - "Alice goes hiking..." (entity link: Alice)
3. Returns ALL Alice memories, properly ranked by relevance

## Why This Architecture?

**Problem with vector-only search**: "Alice loves hiking" and "Alice works at Google" are semantically distant - pure vector search might miss this connection.

**Solution**: Entity links ensure memories about the same person/place/organization are strongly connected regardless of semantic distance.

**Result**: More human-like memory retrieval that understands identity and relationships.

## License

MIT
