"""
Standalone Demo - Cognitive Memory Layer

This example demonstrates the memory system WITHOUT requiring
an external LLM API key. It shows direct API usage with curl-like
requests and manual memory operations.

Useful for:
- Testing the memory system in isolation
- Understanding the API before integrating with LLMs
- Development and debugging

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install httpx:
       pip install httpx
"""

import httpx
import json
from datetime import datetime


# API Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "demo-key-123"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_response(response: httpx.Response, label: str = "Response"):
    """Pretty print an API response."""
    print(f"{label} [{response.status_code}]:")
    try:
        data = response.json()
        print(json.dumps(data, indent=2, default=str))
    except:
        print(response.text)
    print()


def demo_health_check():
    """Check if the API is running."""
    print_section("Health Check")
    
    response = httpx.get(f"{BASE_URL}/health")
    print_response(response, "Health")
    
    if response.status_code == 200:
        print("✓ API is healthy and ready")
        return True
    else:
        print("✗ API is not available. Start it with:")
        print("  docker compose -f docker/docker-compose.yml up api")
        return False


def demo_write_memories():
    """Demonstrate writing different types of memories."""
    print_section("Writing Memories")
    
    user_id = "standalone-demo-user"
    
    # Example memories to store
    memories = [
        {
            "user_id": user_id,
            "content": "User's name is John Smith and he is 35 years old.",
            "memory_type": "semantic_fact",
            "metadata": {"source": "user_introduction"}
        },
        {
            "user_id": user_id,
            "content": "User prefers dark mode in all applications.",
            "memory_type": "preference"
        },
        {
            "user_id": user_id,
            "content": "User is severely allergic to penicillin - this is critical medical information.",
            "memory_type": "constraint"
        },
        {
            "user_id": user_id,
            "content": "On 2024-01-20, user mentioned starting a new job at Google.",
            "memory_type": "episodic_event"
        },
        {
            "user_id": user_id,
            "content": "User seems interested in machine learning based on questions asked.",
            "memory_type": "hypothesis"
        }
    ]
    
    print(f"Storing {len(memories)} memories for user '{user_id}'...\n")
    
    for i, memory in enumerate(memories, 1):
        response = httpx.post(
            f"{BASE_URL}/memory/write",
            headers=HEADERS,
            json=memory
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{i}. ✓ [{memory.get('memory_type', 'auto')}] {memory['content'][:50]}...")
            print(f"   Memory ID: {data.get('memory_id')}")
        else:
            print(f"{i}. ✗ Failed: {response.text}")
    
    return user_id


def demo_read_memories(user_id: str):
    """Demonstrate reading memories with different queries."""
    print_section("Reading Memories")
    
    queries = [
        {
            "query": "What is the user's name?",
            "description": "Simple fact lookup"
        },
        {
            "query": "What are the user's preferences?",
            "description": "Preference retrieval"
        },
        {
            "query": "Are there any medical concerns I should know about?",
            "description": "Constraint lookup (critical info)"
        },
        {
            "query": "Tell me about the user",
            "description": "General retrieval"
        }
    ]
    
    for q in queries:
        print(f"Query: '{q['query']}' ({q['description']})")
        print("-" * 50)
        
        response = httpx.post(
            f"{BASE_URL}/memory/read",
            headers=HEADERS,
            json={
                "user_id": user_id,
                "query": q["query"],
                "max_results": 5,
                "format": "packet"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_count']} memories ({data['elapsed_ms']:.1f}ms)")
            
            for mem in data.get("memories", [])[:3]:
                conf = f"[{mem['confidence']:.0%}]"
                print(f"  • {mem['type']}: {mem['text'][:60]}... {conf}")
        else:
            print(f"Error: {response.text}")
        
        print()


def demo_llm_context_format(user_id: str):
    """Demonstrate the LLM-ready context format."""
    print_section("LLM Context Format")
    
    print("This format is designed for direct injection into LLM prompts.\n")
    
    response = httpx.post(
        f"{BASE_URL}/memory/read",
        headers=HEADERS,
        json={
            "user_id": user_id,
            "query": "Tell me everything about the user",
            "max_results": 10,
            "format": "llm_context"  # This is the key!
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("LLM Context (ready for system prompt):")
        print("-" * 50)
        print(data.get("llm_context", "No context available"))
        print("-" * 50)


def demo_update_memory(user_id: str):
    """Demonstrate updating memories with feedback."""
    print_section("Updating Memories")
    
    # First, get a memory to update
    response = httpx.post(
        f"{BASE_URL}/memory/read",
        headers=HEADERS,
        json={
            "user_id": user_id,
            "query": "machine learning interest",
            "max_results": 1
        }
    )
    
    if response.status_code == 200 and response.json().get("memories"):
        memory = response.json()["memories"][0]
        memory_id = memory["id"]
        
        print(f"Found hypothesis: {memory['text']}")
        print(f"Current confidence: {memory['confidence']:.0%}")
        print()
        
        # Confirm the hypothesis
        print("Confirming this hypothesis (feedback='correct')...")
        
        update_response = httpx.post(
            f"{BASE_URL}/memory/update",
            headers=HEADERS,
            json={
                "user_id": user_id,
                "memory_id": memory_id,
                "feedback": "correct"
            }
        )
        
        if update_response.status_code == 200:
            print(f"✓ Memory updated: {update_response.json()}")
        else:
            print(f"✗ Update failed: {update_response.text}")
    else:
        print("No hypothesis found to update")


def demo_memory_stats(user_id: str):
    """Demonstrate memory statistics."""
    print_section("Memory Statistics")
    
    response = httpx.get(
        f"{BASE_URL}/memory/stats/{user_id}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        stats = response.json()
        print(f"Statistics for user '{user_id}':")
        print(f"  Total memories:  {stats['total_memories']}")
        print(f"  Active memories: {stats['active_memories']}")
        print(f"  Avg confidence:  {stats['avg_confidence']:.0%}")
        print(f"  Avg importance:  {stats['avg_importance']:.0%}")
        print(f"  By type:         {stats['by_type']}")
    else:
        print(f"Error: {response.text}")


def demo_forget_memory(user_id: str):
    """Demonstrate forgetting memories."""
    print_section("Forgetting Memories")
    
    print("Forgetting memories about 'new job'...")
    
    response = httpx.post(
        f"{BASE_URL}/memory/forget",
        headers=HEADERS,
        json={
            "user_id": user_id,
            "query": "new job at Google",
            "action": "archive"  # or "delete" for permanent removal
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Archived {data.get('affected_count', 0)} memories")
    else:
        print(f"✗ Failed: {response.text}")


def demo_curl_examples():
    """Print curl command examples."""
    print_section("Curl Command Examples")
    
    print("You can also use these curl commands directly:\n")
    
    commands = [
        ("Health Check", "curl http://localhost:8000/api/v1/health"),
        ("Write Memory", '''curl -X POST http://localhost:8000/api/v1/memory/write \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: demo-key-123" \\
  -d '{"user_id": "test-user", "content": "User likes pizza"}\''''),
        ("Read Memory", '''curl -X POST http://localhost:8000/api/v1/memory/read \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: demo-key-123" \\
  -d '{"user_id": "test-user", "query": "food preferences", "format": "llm_context"}\''''),
        ("Get Stats", '''curl http://localhost:8000/api/v1/memory/stats/test-user \\
  -H "X-API-Key: demo-key-123"'''),
    ]
    
    for name, cmd in commands:
        print(f"# {name}")
        print(cmd)
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("   Cognitive Memory Layer - Standalone Demo")
    print("=" * 60)
    print("\nThis demo shows the memory API without requiring LLM API keys.")
    print("Press Ctrl+C to skip to the next section.\n")
    
    try:
        # Check health first
        if not demo_health_check():
            return
        
        input("Press Enter to continue with write demo...")
        user_id = demo_write_memories()
        
        input("Press Enter to continue with read demo...")
        demo_read_memories(user_id)
        
        input("Press Enter to continue with LLM context demo...")
        demo_llm_context_format(user_id)
        
        input("Press Enter to continue with update demo...")
        demo_update_memory(user_id)
        
        input("Press Enter to continue with stats demo...")
        demo_memory_stats(user_id)
        
        input("Press Enter to continue with forget demo...")
        demo_forget_memory(user_id)
        
        input("Press Enter to see curl examples...")
        demo_curl_examples()
        
        print_section("Demo Complete!")
        print("You've seen all the core memory operations.")
        print("Check out the other examples for LLM integration:")
        print("  - openai_tool_calling.py")
        print("  - anthropic_tool_calling.py")
        print("  - chatbot_with_memory.py")
        print("  - langchain_integration.py")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except httpx.ConnectError:
        print("\n✗ Could not connect to API. Start it with:")
        print("  docker compose -f docker/docker-compose.yml up api")


if __name__ == "__main__":
    main()
