#!/usr/bin/env python3
"""
Image Tagger - Main Application
CLI and API interface for automated image tagging with GLM-4V + RAG
"""
import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Import local modules
from tagging_chain import create_tagging_chain, ImageTaggingChain


def setup_environment():
    """Setup environment and verify dependencies"""
    # Check for tag database
    tag_db = Path("51標籤庫.json")
    if not tag_db.exists():
        print("❌ Error: Tag database '51標籤庫.json' not found!")
        print("   Please ensure the tag JSON file is in the working directory.")
        sys.exit(1)
    
    print(f"✓ Tag database found: {tag_db}")


def save_results(results: List[dict], output_path: str, format: str = "json"):
    """Save tagging results to file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    elif format == "txt":
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{'='*60}\n")
                f.write(f"Image: {result['image_path']}\n")
                f.write(f"Success: {result['success']}\n")
                
                if result['error']:
                    f.write(f"Error: {result['error']}\n")
                
                if result['tags']:
                    f.write(f"\nTags:\n")
                    for tag in result['tags']:
                        score = result['confidence_scores'].get(tag, 0)
                        f.write(f"  - {tag} ({score:.2f})\n")
                
                f.write("\n")
    
    print(f"\n💾 Results saved to: {output_file}")


def print_results(result: dict, verbose: bool = False):
    """Print tagging results to console"""
    print(f"\n{'='*60}")
    print(f"📋 RESULTS: {Path(result['image_path']).name}")
    print('='*60)
    
    if not result['success']:
        print(f"❌ Error: {result['error']}")
        return
    
    # Tags
    if result['tags']:
        print(f"\n🏷️  Tags Found ({len(result['tags'])}):")
        for i, tag in enumerate(result['tags'][:15], 1):
            score = result['confidence_scores'].get(tag, 0)
            bar = '█' * int(score * 20)
            print(f"  {i:2d}. {tag:20s} [{bar:20s}] {score:.2f}")
        
        if len(result['tags']) > 15:
            print(f"  ... and {len(result['tags']) - 15} more")
    else:
        print("\n⚠️  No tags matched")
    
    # Description (if verbose)
    if verbose and result['description']:
        print(f"\n📝 Description:\n{result['description'][:800]}...")


def cli_tag_single(args):
    """CLI: Tag a single image"""
    setup_environment()
    
    # Validate image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Create chain
    chain = create_tagging_chain(
        vision_base_url=args.vision_url,
        force_reload_tags=args.reload_tags
    )
    
    # Tag image
    result = chain.tag_image(str(image_path))
    
    # Display results
    print_results(result, verbose=args.verbose)
    
    # Save if requested
    if args.output:
        save_results([result], args.output, args.format)


def cli_tag_batch(args):
    """CLI: Tag multiple images"""
    setup_environment()
    
    # Collect image paths
    image_paths = []
    
    for path_str in args.images:
        path = Path(path_str)
        
        if path.is_dir():
            # Find images in directory
            extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            for ext in extensions:
                image_paths.extend(path.glob(f'*{ext}'))
                image_paths.extend(path.glob(f'*{ext.upper()}'))
        elif path.exists():
            image_paths.append(path)
        else:
            print(f"⚠️  Warning: Path not found: {path}")
    
    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print("❌ Error: No valid images found!")
        sys.exit(1)
    
    print(f"\n📁 Found {len(image_paths)} images to process\n")
    
    # Create chain
    chain = create_tagging_chain(
        vision_base_url=args.vision_url,
        force_reload_tags=args.reload_tags
    )
    
    # Process images
    results = []
    failed = []
    
    def progress(current, total):
        print(f"\n📊 Progress: [{current}/{total}] {current/total*100:.1f}%")
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path.name}")
        
        result = chain.tag_image(str(img_path))
        results.append(result)
        
        if not result['success']:
            failed.append((img_path.name, result['error']))
        
        print_results(result, verbose=args.verbose)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE")
    print('='*60)
    print(f"Total: {len(results)}")
    print(f"Successful: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\n❌ Failed items:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    # Save results
    if args.output:
        save_results(results, args.output, args.format)


def cli_init_db(args):
    """CLI: Initialize tag database"""
    from tag_vector_store import init_tag_store
    
    print("🗄️  Initializing Tag Database\n")
    
    store = init_tag_store(
        json_path=args.json_path or "51標籤庫.json",
        persist_directory=args.persist_dir,
        force_reload=args.force
    )
    
    stats = store.get_collection_stats()
    print(f"\n{'='*60}")
    print("✅ Database Initialized")
    print('='*60)
    print(f"Collection: {stats['collection_name']}")
    print(f"Total Tags: {stats['total_tags']}")
    print(f"Storage: {stats['persist_directory']}")


def cli_search_tags(args):
    """CLI: Search tags by query"""
    from tag_vector_store import TagVectorStore
    
    store = TagVectorStore(persist_directory=args.persist_dir)
    
    print(f"🔍 Searching: '{args.query}'\n")
    
    results = store.search(
        query=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )
    
    if not results:
        print("❌ No matching tags found")
        return
    
    print(f"Found {len(results)} matches:\n")
    
    for i, tag in enumerate(results, 1):
        bar = '█' * int(tag['similarity'] * 20)
        print(f"{i:2d}. {tag['tag_name']:25s} [{bar:20s}] {tag['similarity']:.3f}")
        if args.verbose:
            print(f"    {tag['description'][:100]}...")
            print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🖼️  Image Tagger - GLM-4V + RAG Tagging System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tag a single image
  python tagger.py single image.jpg
  
  # Tag with verbose output
  python tagger.py single image.jpg -v
  
  # Tag batch of images
  python tagger.py batch ./images/ -o results.json
  
  # Initialize tag database
  python tagger.py init-db --force
  
  # Search tags
  python tagger.py search "cat girl" -k 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--vision-url',
        default='http://localhost:1234/v1',
        help='LM Studio API endpoint (default: http://localhost:1234/v1)'
    )
    parent_parser.add_argument(
        '--persist-dir',
        default='./chroma_db',
        help='ChromaDB persistence directory'
    )
    
    # Single image command
    single_parser = subparsers.add_parser(
        'single',
        parents=[parent_parser],
        help='Tag a single image'
    )
    single_parser.add_argument('image', help='Path to image file')
    single_parser.add_argument('-o', '--output', help='Output file path')
    single_parser.add_argument('-f', '--format', choices=['json', 'txt'], default='json')
    single_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    single_parser.add_argument('--reload-tags', action='store_true', help='Force reload tag embeddings')
    
    # Batch command
    batch_parser = subparsers.add_parser(
        'batch',
        parents=[parent_parser],
        help='Tag multiple images'
    )
    batch_parser.add_argument('images', nargs='+', help='Image files or directories')
    batch_parser.add_argument('-o', '--output', help='Output file path', required=True)
    batch_parser.add_argument('-f', '--format', choices=['json', 'txt'], default='json')
    batch_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    batch_parser.add_argument('--reload-tags', action='store_true', help='Force reload tag embeddings')
    
    # Init DB command
    init_parser = subparsers.add_parser(
        'init-db',
        parents=[parent_parser],
        help='Initialize tag database'
    )
    init_parser.add_argument('--json-path', help='Path to tag JSON file')
    init_parser.add_argument('--force', action='store_true', help='Force reload')
    
    # Search command
    search_parser = subparsers.add_parser(
        'search',
        parents=[parent_parser],
        help='Search tags by query'
    )
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', '--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('-t', '--threshold', type=float, default=0.0, help='Similarity threshold')
    search_parser.add_argument('-v', '--verbose', action='store_true', help='Show descriptions')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate handler
    if args.command == 'single':
        cli_tag_single(args)
    elif args.command == 'batch':
        cli_tag_batch(args)
    elif args.command == 'init-db':
        cli_init_db(args)
    elif args.command == 'search':
        cli_search_tags(args)


if __name__ == '__main__':
    main()
