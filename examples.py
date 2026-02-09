#!/usr/bin/env python3
"""
Quick Start Example
Demonstrates basic usage of the image tagging system
"""
from tagging_chain import create_tagging_chain
from tag_vector_store import init_tag_store


def example_1_basic_tagging():
    """Example: Tag a single image"""
    print("="*60)
    print("Example 1: Basic Image Tagging")
    print("="*60)
    
    # Create tagging chain
    chain = create_tagging_chain()
    
    # Tag an image
    result = chain.tag_image("path/to/your/image.jpg")
    
    # Print results
    print(f"\nTags found: {len(result['tags'])}")
    for tag in result['tags'][:10]:
        score = result['confidence_scores'][tag]
        print(f"  - {tag}: {score:.2f}")


def example_2_batch_processing():
    """Example: Process multiple images"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    chain = create_tagging_chain()
    
    # List of image paths
    images = [
        "image1.jpg",
        "image2.png",
        "image3.jpg"
    ]
    
    # Process all images
    results = chain.tag_images_batch(images)
    
    # Summary
    for r in results:
        print(f"{r['image_path']}: {len(r['tags'])} tags")


def example_3_search_tags():
    """Example: Search tags by description"""
    print("\n" + "="*60)
    print("Example 3: Tag Search")
    print("="*60)
    
    # Initialize tag store
    store = init_tag_store()
    
    # Search for tags
    query = "character with animal features"
    results = store.search(query, top_k=5)
    
    print(f"\nQuery: '{query}'\n")
    for tag in results:
        print(f"  {tag['tag_name']} ({tag['similarity']:.2f})")


def example_4_custom_prompt():
    """Example: Use custom analysis prompt"""
    from glm4v_client import create_vision_client
    
    print("\n" + "="*60)
    print("Example 4: Custom Vision Prompt")
    print("="*60)
    
    client = create_vision_client()
    
    # Custom prompt for specific analysis
    custom_prompt = """
    Analyze this image and focus on:
    1. Clothing details and style
    2. Physical appearance and features
    3. Any accessories or props
    Provide a concise description.
    """
    
    result = client.analyze_image_sync(
        image_path="image.jpg",
        prompt=custom_prompt
    )
    
    if result['success']:
        print(result['description'])


def example_5_integration():
    """Example: Integrate into your application"""
    print("\n" + "="*60)
    print("Example 5: Application Integration")
    print("="*60)
    
    # Initialize once
    chain = create_tagging_chain()
    
    # Use in your application
    def process_uploaded_image(image_path: str) -> dict:
        """Process uploaded image and return tags"""
        result = chain.tag_image(image_path)
        
        return {
            'success': result['success'],
            'tags': result['tags'],
            'confidence': result['confidence_scores'],
            'description': result['description'][:200] if result['description'] else None
        }
    
    # Example usage
    response = process_uploaded_image("uploaded_image.jpg")
    print(f"Processed: {response}")


if __name__ == "__main__":
    print("""
Image Tagger - Usage Examples
==============================

This script demonstrates various ways to use the image tagging system.

Prerequisites:
1. Install dependencies: pip install -r requirements.txt
2. Start LM Studio with GLM-4V model on port 1234
3. Initialize tag database: python tagger.py init-db

Choose an example to run:
    """)
    
    examples = [
        ("Basic tagging", example_1_basic_tagging),
        ("Batch processing", example_2_batch_processing),
        ("Search tags", example_3_search_tags),
        ("Custom prompt", example_4_custom_prompt),
        ("Integration", example_5_integration),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nRun: python examples.py")
