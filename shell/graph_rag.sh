#!/bin/bash
# =====================================================
# Graph RAG Automation Script
# =====================================================
# Using KGARevion Agent (https://arxiv.org/abs/2410.04660)
# 
# Core Actions:
#   1. Generate - Extract medical concepts and generate triplets
#   2. Review   - Verify triplets against MDKG
#   3. Revise   - Iteratively correct rejected triplets  
#   4. Answer   - Generate final answer based on verified knowledge
#
# Optimization: Community Detection (Leiden/Louvain algorithm)
# Reference: GraphRAG paper (https://arxiv.org/abs/2404.16130)
# =====================================================

set -e

# Default values
LLM_TYPE="openai"
LLM_MODEL="gpt-4"
MODE="interactive"
QUERY=""
VERBOSE=""
USE_COMMUNITY=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Usage function
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive    Run in interactive Q&A mode"
    echo "  -d, --demo           Run demo mode with sample questions"
    echo "  -q, --query <text>   Run single query"
    echo "  -v, --verbose        Enable verbose output"
    echo "  -c, --community      Enable community detection optimization"
    echo "  --no-community       Disable community detection (default)"
    echo "  --llm <type>         LLM backend: openai or ollama (default: openai)"
    echo "  --model <name>       LLM model name (default: gpt-4)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Interactive mode with OpenAI"
    echo "  bash $0 -i"
    echo ""
    echo "  # Demo mode with verbose output"
    echo "  bash $0 -d -v"
    echo ""
    echo "  # Single query with community detection"
    echo "  bash $0 -q \"What are the symptoms of depression?\" -c"
    echo ""
    echo "  # Use local Ollama"
    echo "  bash $0 --llm ollama --model llama3 -i"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY       Required for OpenAI backend"
    echo "  OLLAMA_HOST          Optional Ollama host (default: http://localhost:11434)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            MODE="interactive"
            shift
            ;;
        -d|--demo)
            MODE="demo"
            shift
            ;;
        -q|--query)
            MODE="single"
            QUERY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -c|--community)
            USE_COMMUNITY="--community"
            shift
            ;;
        --no-community)
            USE_COMMUNITY="--no-community"
            shift
            ;;
        --llm)
            LLM_TYPE="$2"
            shift 2
            ;;
        --model)
            LLM_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check for API key if using OpenAI
if [[ "$LLM_TYPE" == "openai" && -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Change to project directory
cd "$PROJECT_ROOT"

echo "=============================================="
echo "   MDKG Graph RAG - KGARevion Implementation"
echo "=============================================="
echo "LLM Backend: $LLM_TYPE"
echo "Model: $LLM_MODEL"
echo "Mode: $MODE"
if [[ -n "$USE_COMMUNITY" ]]; then
    echo "Community Detection: $USE_COMMUNITY"
fi
echo "=============================================="

# Build command
CMD="python GraphRAG/graph_rag_demo.py"
CMD="$CMD --llm $LLM_TYPE"
CMD="$CMD --model $LLM_MODEL"

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

if [[ -n "$USE_COMMUNITY" ]]; then
    CMD="$CMD $USE_COMMUNITY"
fi

case $MODE in
    interactive)
        CMD="$CMD --interactive"
        ;;
    demo)
        CMD="$CMD --demo"
        ;;
    single)
        if [[ -z "$QUERY" ]]; then
            echo "Error: Query text is required for single mode"
            exit 1
        fi
        CMD="$CMD --query \"$QUERY\""
        ;;
esac

# Execute
echo "Executing: $CMD"
echo ""
eval $CMD
