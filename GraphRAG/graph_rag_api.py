"""
Graph RAG API Server for Mental Disorder Knowledge Graph

This module provides a REST API interface for the Graph RAG system,
allowing integration with web applications and other services.

Usage:
    python graph_rag_api.py --port 8000
    
API Endpoints:
    POST /query      - Submit a mental health query
    GET /health      - Health check
    GET /info        - System information
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, Dict, Any
from dataclasses import asdict

# Add project root to path (parent directory of GraphRAG folder)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import FastAPI, fall back to Flask if not available
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    USE_FASTAPI = True
except ImportError:
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        USE_FASTAPI = False
    except ImportError:
        print("Please install either FastAPI or Flask:")
        print("  pip install fastapi uvicorn")
        print("  or")
        print("  pip install flask flask-cors")
        sys.exit(1)

from graph_rag import create_graph_rag, MentalDisorderGraphRAG, QueryResult


# Global RAG instance
rag_instance: Optional[MentalDisorderGraphRAG] = None


def initialize_rag(llm_type: str = "openai", llm_model: str = "gpt-4", 
                   api_key: Optional[str] = None, **kwargs) -> MentalDisorderGraphRAG:
    """Initialize the Graph RAG system"""
    global rag_instance
    
    logger.info(f"Initializing Graph RAG with {llm_type}/{llm_model}")
    
    rag_instance = create_graph_rag(
        llm_type=llm_type,
        llm_model=llm_model,
        api_key=api_key,
        **kwargs
    )
    
    logger.info("Graph RAG initialized successfully")
    return rag_instance


def query_result_to_dict(result: QueryResult) -> Dict[str, Any]:
    """Convert QueryResult to dictionary for JSON response"""
    return {
        "query": result.query,
        "answer": result.answer,
        "confidence": result.confidence,
        "medical_concepts": result.medical_concepts,
        "verified_triplets": [t.to_dict() for t in result.verified_triplets],
        "rejected_triplets": [t.to_dict() for t in result.rejected_triplets],
        "revised_triplets": [t.to_dict() for t in result.revised_triplets],
        "reasoning_trace": result.reasoning_trace
    }


# ============ FastAPI Implementation ============
if USE_FASTAPI:
    
    app = FastAPI(
        title="Mental Disorder Graph RAG API",
        description="Knowledge Graph-based RAG for Mental Health Questions",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class QueryRequest(BaseModel):
        question: str
        verbose: bool = False
    
    class QueryResponse(BaseModel):
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
    
    class HealthResponse(BaseModel):
        status: str
        rag_initialized: bool
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            rag_initialized=rag_instance is not None
        )
    
    @app.get("/info")
    async def system_info():
        """Get system information"""
        info = {
            "name": "Mental Disorder Graph RAG",
            "version": "1.0.0",
            "rag_initialized": rag_instance is not None,
            "knowledge_graph": {
                "entities": len(rag_instance.kg.entities) if rag_instance else 0,
                "triplets": len(rag_instance.kg.triplets) if rag_instance else 0
            } if rag_instance else None
        }
        return info
    
    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest):
        """Process a mental health query"""
        if rag_instance is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        try:
            result = rag_instance.query(request.question, verbose=request.verbose)
            return QueryResponse(
                success=True,
                data=query_result_to_dict(result)
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                success=False,
                error=str(e)
            )
    
    @app.post("/batch_query")
    async def batch_query(questions: list[str]):
        """Process multiple queries"""
        if rag_instance is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        results = []
        for question in questions:
            try:
                result = rag_instance.query(question)
                results.append({
                    "question": question,
                    "success": True,
                    "data": query_result_to_dict(result)
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
        
        return {"results": results}
    
    def run_server(host: str, port: int):
        """Run the FastAPI server"""
        uvicorn.run(app, host=host, port=port)


# ============ Flask Implementation ============
else:
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "rag_initialized": rag_instance is not None
        })
    
    @app.route("/info", methods=["GET"])
    def system_info():
        """Get system information"""
        info = {
            "name": "Mental Disorder Graph RAG",
            "version": "1.0.0",
            "rag_initialized": rag_instance is not None,
            "knowledge_graph": {
                "entities": len(rag_instance.kg.entities) if rag_instance else 0,
                "triplets": len(rag_instance.kg.triplets) if rag_instance else 0
            } if rag_instance else None
        }
        return jsonify(info)
    
    @app.route("/query", methods=["POST"])
    def process_query():
        """Process a mental health query"""
        if rag_instance is None:
            return jsonify({"success": False, "error": "RAG system not initialized"}), 503
        
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"success": False, "error": "Missing 'question' field"}), 400
        
        try:
            verbose = data.get("verbose", False)
            result = rag_instance.query(data["question"], verbose=verbose)
            return jsonify({
                "success": True,
                "data": query_result_to_dict(result)
            })
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route("/batch_query", methods=["POST"])
    def batch_query():
        """Process multiple queries"""
        if rag_instance is None:
            return jsonify({"success": False, "error": "RAG system not initialized"}), 503
        
        data = request.get_json()
        if not data or "questions" not in data:
            return jsonify({"success": False, "error": "Missing 'questions' field"}), 400
        
        results = []
        for question in data["questions"]:
            try:
                result = rag_instance.query(question)
                results.append({
                    "question": question,
                    "success": True,
                    "data": query_result_to_dict(result)
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({"results": results})
    
    def run_server(host: str, port: int):
        """Run the Flask server"""
        app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Graph RAG API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--llm", type=str, default="openai", 
                       choices=["openai", "ollama", "huggingface"],
                       help="LLM backend to use")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("  Mental Disorder Graph RAG API Server")
    print(f"  Backend: {'FastAPI' if USE_FASTAPI else 'Flask'}")
    print(f"{'='*60}\n")
    
    # Initialize RAG
    try:
        initialize_rag(
            llm_type=args.llm,
            llm_model=args.model,
            api_key=args.api_key,
            ollama_url=args.ollama_url
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    print(f"\n📡 API available at: http://{args.host}:{args.port}")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs" if USE_FASTAPI else "")
    print("\nEndpoints:")
    print("  POST /query       - Submit a query")
    print("  GET  /health      - Health check")
    print("  GET  /info        - System info")
    print("\n")
    
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
