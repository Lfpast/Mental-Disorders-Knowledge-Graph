#!/bin/bash
# Graph RAG Shell Script for Mental Disorder Knowledge Graph
# This script provides easy access to the Graph RAG functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Mental Disorder Knowledge Graph RAG${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not found${NC}"
    exit 1
fi

# Check for required packages
echo -e "${YELLOW}Checking dependencies...${NC}"

python3 -c "import torch" 2>/dev/null || {
    echo -e "${RED}Missing: torch - Please run: pip install torch${NC}"
    exit 1
}

python3 -c "import transformers" 2>/dev/null || {
    echo -e "${RED}Missing: transformers - Please run: pip install transformers${NC}"
    exit 1
}

python3 -c "import openai" 2>/dev/null || {
    echo -e "${YELLOW}Warning: openai not installed. Install with: pip install openai${NC}"
    echo -e "${YELLOW}You can still use Ollama backend with --llm ollama${NC}"
}

echo -e "${GREEN}Dependencies check passed!${NC}"
echo ""

# Parse arguments
MODE="interactive"
LLM="openai"
MODEL="gpt-4"
VERBOSE=""
QUERY=""

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive     Run in interactive mode (default)"
    echo "  -d, --demo            Run demo with sample questions"
    echo "  -q, --query TEXT      Process a single query"
    echo "  --llm TYPE            LLM backend: openai, ollama, huggingface (default: openai)"
    echo "  --model NAME          Model name (default: gpt-4)"
    echo "  -v, --verbose         Enable verbose output"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i                                    # Interactive mode with OpenAI"
    echo "  $0 --llm ollama --model llama3 -i       # Interactive with local Ollama"
    echo "  $0 -q 'What are symptoms of depression?' # Single query"
    echo "  $0 -d -v                                 # Demo mode with verbose output"
}

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
            MODE="query"
            QUERY="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check API key for OpenAI
if [ "$LLM" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Note: OPENAI_API_KEY environment variable not set${NC}"
    echo -e "${YELLOW}You will be prompted for the API key, or use --llm ollama${NC}"
    echo ""
fi

# Run the Graph RAG demo
echo -e "${GREEN}Starting Graph RAG...${NC}"
echo ""

case $MODE in
    interactive)
        python3 "$PROJECT_ROOT/GraphRAG/graph_rag_demo.py" --interactive --llm "$LLM" --model "$MODEL" $VERBOSE
        ;;
    demo)
        python3 "$PROJECT_ROOT/GraphRAG/graph_rag_demo.py" --demo --llm "$LLM" --model "$MODEL" $VERBOSE
        ;;
    query)
        python3 "$PROJECT_ROOT/GraphRAG/graph_rag_demo.py" --query "$QUERY" --llm "$LLM" --model "$MODEL" $VERBOSE
        ;;
esac

echo ""
echo -e "${GREEN}Graph RAG session ended${NC}"
