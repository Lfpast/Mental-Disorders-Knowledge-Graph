#!/usr/bin/env python3
"""
Graph RAG Demo Script for Mental Disorder Knowledge Graph

This script demonstrates the KGARevion-style Graph RAG functionality for answering
mental health related questions using the MDKG knowledge graph.

Based on KGARevion paper: https://arxiv.org/abs/2410.04660

Usage:
    python graph_rag_demo.py --interactive
    python graph_rag_demo.py --query "What are the symptoms of depression?"
    python graph_rag_demo.py --demo
"""

import os
import sys
import argparse
import json
from typing import Optional

# Add project root to path (parent directory of GraphRAG folder)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kgarevion_agent import create_kgarevion_agent, KGARevionAgent, QueryResult


# Sample questions for demonstration
DEMO_QUESTIONS = [
    "What are the main symptoms of major depressive disorder?",
    "How does lithium work in treating bipolar disorder?",
    "What is the relationship between anxiety and serotonin?",
    "What are the risk factors for schizophrenia?",
    "How does cognitive behavioral therapy help with OCD?",
    "What are the common comorbidities of ADHD?",
    "Explain the connection between sleep disorders and depression.",
    "What neurotransmitters are involved in anxiety disorders?",
]


def print_banner():
    """Print a welcome banner"""
    print("\n" + "="*70)
    print("   Mental Disorder Knowledge Graph - Graph RAG Demo")
    print("   Powered by KGARevion (arxiv.org/abs/2410.04660)")
    print("="*70 + "\n")


def print_result(result: QueryResult, verbose: bool = False):
    """Pretty print a query result"""
    print("\n" + "-"*60)
    print("📋 ANSWER:")
    print("-"*60)
    print(result.answer)
    print("-"*60)
    
    # Count triplets for summary
    true_count = len(result.true_triplets)
    false_count = len(result.false_triplets)
    incomplete_count = len(result.incomplete_triplets)
    
    print(f"📊 Triplet Statistics: ✅ {true_count} TRUE | ❌ {false_count} FALSE | ❓ {incomplete_count} INCOMPLETE")
    
    if verbose:
        print(f"\n🔬 Medical Concepts: {', '.join(result.medical_concepts) if result.medical_concepts else 'None'}")
        
        if result.true_triplets:
            print("\n📝 Verified TRUE Triplets:")
            for t in result.true_triplets[:5]:
                print(f"   ✅ ({t.head_entity}) --[{t.relation}]--> ({t.tail_entity})")
        
        if result.false_triplets:
            print("\n📝 FALSE Triplets (revised):")
            for t in result.false_triplets[:3]:
                print(f"   ❌ ({t.head_entity}) --[{t.relation}]--> ({t.tail_entity})")
        
        if result.reasoning_trace:
            print("\n🧠 Reasoning Trace:")
            for step in result.reasoning_trace:
                print(f"   → {step}")
    
    print()


def run_interactive_mode(rag: KGARevionAgent, verbose: bool = False):
    """Run interactive Q&A session"""
    print("\n💬 Interactive Mode - Ask questions about mental health")
    print("   Type 'quit' or 'exit' to stop, 'help' for sample questions\n")
    
    while True:
        try:
            question = input("🔍 Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if question.lower() == 'help':
            print("\n📚 Sample Questions:")
            for i, q in enumerate(DEMO_QUESTIONS, 1):
                print(f"   {i}. {q}")
            print()
            continue
        
        # Check if user entered a number for sample questions
        if question.isdigit():
            idx = int(question) - 1
            if 0 <= idx < len(DEMO_QUESTIONS):
                question = DEMO_QUESTIONS[idx]
                print(f"   → {question}")
            else:
                print("   Invalid number. Type 'help' for sample questions.")
                continue
        
        print("\n⏳ Processing with KGARevion pipeline...\n")
        
        try:
            result = rag.query(question, verbose=verbose)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")


def run_demo_mode(rag: KGARevionAgent, num_questions: int = 3, verbose: bool = False):
    """Run a demonstration with sample questions"""
    print("\n🎯 Demo Mode - Processing sample questions...\n")
    
    for i, question in enumerate(DEMO_QUESTIONS[:num_questions], 1):
        print(f"\n{'='*60}")
        print(f"📌 Question {i}/{num_questions}:")
        print(f"   \"{question}\"")
        print("="*60)
        
        try:
            result = rag.query(question, verbose=verbose)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
        
        if i < num_questions:
            print("\n" + "·"*60 + "\n")


def run_single_query(rag: KGARevionAgent, query: str, verbose: bool = False, 
                     output_file: Optional[str] = None):
    """Run a single query"""
    print(f"\n📌 Query: \"{query}\"")
    print("⏳ Processing with KGARevion pipeline...\n")
    
    result = rag.query(query, verbose=verbose)
    print_result(result, verbose=verbose)
    
    # Save to file if requested
    if output_file:
        output_data = {
            "query": result.query,
            "answer": result.answer,
            "medical_concepts": result.medical_concepts,
            "true_triplets": [t.to_dict() for t in result.true_triplets],
            "false_triplets": [t.to_dict() for t in result.false_triplets],
            "incomplete_triplets": [t.to_dict() for t in result.incomplete_triplets],
            "revised_triplets": [t.to_dict() for t in result.revised_triplets],
            "reasoning_trace": result.reasoning_trace
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Mental Disorder Graph RAG Demo (KGARevion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive              # Interactive Q&A mode
  %(prog)s --demo                     # Run demo with sample questions
  %(prog)s -q "symptoms of depression" # Single query
  %(prog)s -q "..." --llm ollama --model llama3  # Use local Ollama
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", "-i", action="store_true",
                           help="Run in interactive mode")
    mode_group.add_argument("--demo", "-d", action="store_true",
                           help="Run demonstration with sample questions")
    mode_group.add_argument("--query", "-q", type=str,
                           help="Single query to process")
    
    # LLM configuration
    parser.add_argument("--llm", type=str, default="openai",
                       choices=["openai", "ollama"],
                       help="LLM backend to use (default: openai)")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Model name (default: gpt-4)")
    parser.add_argument("--api-key", type=str,
                       help="API key for OpenAI (or set OPENAI_API_KEY env)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                       help="Ollama server URL")
    
    # Community Detection
    parser.add_argument("--no-community", action="store_true",
                       help="Disable community detection optimization")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output with triplet details")
    parser.add_argument("--output", "-o", type=str,
                       help="Save results to JSON file")
    parser.add_argument("--num-demo", type=int, default=3,
                       help="Number of demo questions to run")
    
    # Configuration
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check for required configuration
    if args.llm == "openai" and not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key required!")
        print("   Set OPENAI_API_KEY environment variable or use --api-key")
        print("   Alternatively, use --llm ollama for local LLM")
        sys.exit(1)
    
    # Initialize Graph RAG
    print("🔧 Initializing KGARevion Agent...")
    print(f"   LLM Backend: {args.llm}")
    print(f"   Model: {args.model}")
    print(f"   Community Detection: {'Disabled' if args.no_community else 'Enabled (Leiden/Louvain)'}")
    
    try:
        rag = create_kgarevion_agent(
            config_path=args.config,
            llm_type=args.llm,
            llm_model=args.model,
            api_key=args.api_key,
            use_community_detection=not args.no_community,
            ollama_url=args.ollama_url
        )
        print("✅ KGARevion Agent initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize KGARevion Agent: {e}")
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.interactive:
            run_interactive_mode(rag, verbose=args.verbose)
        elif args.demo:
            run_demo_mode(rag, num_questions=args.num_demo, verbose=args.verbose)
        elif args.query:
            run_single_query(rag, args.query, verbose=args.verbose, output_file=args.output)
        else:
            # Default to interactive mode
            run_interactive_mode(rag, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
