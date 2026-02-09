#!/usr/bin/env python3
"""
Simple demonstration of the Manga Tagger system.
This script shows how to use both the CLI and API interfaces.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

def demo_cli_mode():
    """Demonstrate CLI usage."""
    print("=== CLI Mode Demo ===")
    print("1. Check available commands:")
    os.system("python tagger.py --help")
    print("\n2. Search for tags:")
    os.system("python tagger.py search \"cat girl\" -k 3")
    
    print("\n3. Initialize database (if needed):")
    print("python tagger.py init-db --force")

def demo_api_mode():
    """Demonstrate API usage."""
    print("=== API Mode Demo ===")
    print("1. Starting server in background...")
    
    # Start server in background
    import subprocess
    import requests
    
    try:
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0", "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        print("   Waiting for server to start...")
        for i in range(10):
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    print(f"   Server started! ({i+1} seconds)")
                    break
            except:
                time.sleep(1)
        else:
            print("   Failed to start server")
            server_process.terminate()
            return
        
        print("\n2. Testing API endpoints:")
        
        # Health check
        print("   Health check:")
        response = requests.get("http://localhost:8000/health")
        print(f"   Status: {response.status_code}")
        health_data = response.json()
        print(f"   Mock mode: {health_data['models_loaded'].get('mock_mode', 'unknown')}")
        
        # Tags list
        print("\n   Tags list:")
        response = requests.get("http://localhost:8000/tags")
        tags_data = response.json()
        print(f"   Total tags available: {tags_data.get('total', 0)}")
        
        if tags_data.get('tags'):
            print("   Sample tags:")
            for tag in tags_data['tags'][:3]:
                print(f"     - {tag['tag_name']}")
        
        # RAG stats
        print("\n   RAG statistics:")
        response = requests.get("http://localhost:8000/rag/stats")
        rag_data = response.json()
        print(f"   Documents in database: {rag_data.get('total_documents', 0)}")
        
        # Tag cover (mock image)
        print("\n   Tag cover (mock test):")
        mock_image = b"fake_image_data"
        files = {'file': ('test.jpg', mock_image, 'image/jpeg')}
        data = {'top_k': 3, 'confidence_threshold': 0.5}
        
        response = requests.post("http://localhost:8000/tag-cover", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            print(f"   Generated {len(result.get('tags', []))} tags:")
            for tag in result.get('tags', []):
                print(f"     - {tag['tag']} (confidence: {tag['confidence']:.2f})")
        else:
            print(f"   Error: {response.status_code}")
        
        print("\n3. API Documentation:")
        print("   Swagger UI: http://localhost:8000/docs")
        print("   ReDoc: http://localhost:8000/redoc")
        
        print("\n   Press Ctrl+C to stop server...")
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("   Stopping server...")
            server_process.terminate()
            server_process.wait()
            
    except Exception as e:
        print(f"Error running demo: {e}")

def show_system_info():
    """Show system information."""
    print("=== System Information ===")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Check dependencies
    dependencies = [
        "fastapi", "uvicorn", "pydantic", 
        "torch", "transformers", "chromadb"
    ]
    
    print("\nDependencies check:")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  [OK] {dep}")
        except ImportError:
            print(f"  [MISSING] {dep} (not installed)")
    
    # Check files
    print("\nProject files:")
    important_files = [
        "app/main.py", "app/api/routes.py", 
        "data/tags.json", ".env", "requirements.txt"
    ]
    
    for file_path in important_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path} (missing)")
    
    # Check directories
    print("\nData directories:")
    dirs = ["data/chroma_db", "data/rag_dataset"]
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [MISSING] {dir_path} (missing)")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Manga Tagger Demo")
    parser.add_argument(
        "--mode", 
        choices=["cli", "api", "info", "all"], 
        default="all",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("Manga Cover Auto-Tagger - Demo")
    print("=" * 40)
    
    if args.mode in ["info", "all"]:
        show_system_info()
        print()
    
    if args.mode in ["cli", "all"]:
        demo_cli_mode()
        print()
    
    if args.mode in ["api", "all"]:
        demo_api_mode()
        print()
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Prepare manga cover images")
    print("2. Add reference images to RAG dataset")
    print("3. Configure real models for production use")
    print("4. Customize tag library for your needs")

if __name__ == "__main__":
    main()