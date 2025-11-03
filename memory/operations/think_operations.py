"""
Think operations for formulating answers based on agent and world facts.
"""

import os
from typing import Dict, List, Any
from openai import AsyncOpenAI


class ThinkOperationsMixin:
    """Mixin class for think operations."""

    async def think_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Think and formulate an answer using agent identity and world facts.

        This method:
        1. Retrieves agent facts (agent's identity and past actions)
        2. Retrieves world facts (general knowledge)
        3. Uses Groq LLM to formulate an answer
        4. Returns plain text answer and the facts used

        Args:
            agent_id: Agent identifier
            query: Question to answer
            thinking_budget: Number of memory units to explore
            top_k: Maximum facts to retrieve
            model: LLM model to use (default: llama-3.3-70b-versatile)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dict with:
                - text: Plain text answer (no markdown)
                - based_on: Dict with 'world' and 'agent' fact lists
        """
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        client = AsyncOpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Step 1: Get agent facts (identity)
        agent_results, _ = await self.search_async(
            agent_id=agent_id,
            query=query,
            thinking_budget=thinking_budget,
            top_k=top_k,
            enable_trace=False,
            fact_type='agent'
        )

        # Step 2: Get world facts
        world_results, _ = await self.search_async(
            agent_id=agent_id,
            query=query,
            thinking_budget=thinking_budget,
            top_k=top_k,
            enable_trace=False,
            fact_type='world'
        )

        # Step 3: Format facts for LLM
        agent_facts_text = "\n".join([f"- {fact['text']}" for fact in agent_results]) if agent_results else "None"
        world_facts_text = "\n".join([f"- {fact['text']}" for fact in world_results]) if world_results else "None"

        # Step 4: Call Groq to formulate answer
        prompt = f"""You are an AI assistant answering a question based on retrieved facts.

AGENT IDENTITY (what the agent has done):
{agent_facts_text}

WORLD FACTS (general knowledge):
{world_facts_text}

QUESTION: {query}

Provide a helpful, accurate answer based on the facts above. If the facts don't contain enough information to answer the question, say so clearly. Do not use markdown formatting - respond in plain text only."""

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Always respond in plain text without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        answer_text = response.choices[0].message.content.strip()

        # Step 5: Return response with facts split by type
        return {
            "text": answer_text,
            "based_on": {
                "world": world_results,
                "agent": agent_results
            }
        }
