"""
Image Tagging Chain using LangGraph
Orchestrates the workflow: Image -> Vision Analysis -> RAG Tag Retrieval -> Results
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pathlib import Path
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from tag_vector_store import TagVectorStore
from glm4v_client import GLM4VisionClient, ImageTaggingPrompts


# Define state schema
class TaggingState(TypedDict):
    """State for the image tagging workflow"""
    image_path: str
    image_description: Optional[str]
    vision_error: Optional[str]
    retrieved_tags: List[Dict[str, Any]]
    final_tags: List[str]
    confidence_scores: Dict[str, float]
    processing_complete: bool
    metadata: Dict[str, Any]


class ImageTaggingChain:
    """
    LangGraph-based chain for automated image tagging
    
    Workflow:
    1. Vision Analysis: GLM-4V analyzes image and generates description
    2. Tag Retrieval: RAG retrieves relevant tags based on description
    3. Tag Ranking: Rank and filter tags by relevance
    4. Output: Return final tag list with confidence scores
    """
    
    def __init__(
        self,
        tag_store: TagVectorStore,
        vision_client: GLM4VisionClient,
        top_k_tags: int = 15,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize the tagging chain
        
        Args:
            tag_store: Initialized TagVectorStore
            vision_client: GLM4VisionClient for image analysis
            top_k_tags: Number of tags to retrieve per query
            similarity_threshold: Minimum similarity for tag inclusion
        """
        self.tag_store = tag_store
        self.vision_client = vision_client
        self.top_k_tags = top_k_tags
        self.similarity_threshold = similarity_threshold
        self.prompts = ImageTaggingPrompts()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _analyze_image_node(self, state: TaggingState) -> TaggingState:
        """
        Node: Analyze image using GLM-4V
        """
        try:
            print(f"🔍 Analyzing image: {state['image_path']}")
            
            result = asyncio.run(self.vision_client.analyze_image(
                image_path=state['image_path'],
                prompt=self.prompts.get_tagging_prompt(),
                system_prompt=self.prompts.get_system_prompt()
            ))
            
            if result['success']:
                state['image_description'] = result['description']
                state['metadata']['vision_model'] = result.get('model', 'unknown')
                state['metadata']['token_usage'] = result.get('usage', {})
                print(f"  ✓ Analysis complete ({len(result['description'])} chars)")
            else:
                state['vision_error'] = result.get('error', 'Unknown error')
                print(f"  ✗ Analysis failed: {state['vision_error']}")
                
        except Exception as e:
            state['vision_error'] = str(e)
            print(f"  ✗ Analysis error: {e}")
        
        return state
    
    def _retrieve_tags_node(self, state: TaggingState) -> TaggingState:
        """
        Node: Retrieve relevant tags using RAG
        """
        if state['vision_error']:
            print("⚠ Skipping tag retrieval due to vision error")
            return state
        
        description = state['image_description']
        if not description:
            print("⚠ No description available for tag retrieval")
            return state
        
        print(f"🔎 Retrieving tags from vector store...")
        
        try:
            # Search for relevant tags
            results = self.tag_store.search(
                query=description,
                top_k=self.top_k_tags,
                similarity_threshold=self.similarity_threshold
            )
            
            state['retrieved_tags'] = results
            print(f"  ✓ Retrieved {len(results)} tags")
            
            # Show top matches
            for i, tag in enumerate(results[:5], 1):
                print(f"    {i}. {tag['tag_name']} ({tag['similarity']:.2f})")
                
        except Exception as e:
            print(f"  ✗ Tag retrieval error: {e}")
            state['vision_error'] = f"Tag retrieval failed: {e}"
        
        return state
    
    def _rank_tags_node(self, state: TaggingState) -> TaggingState:
        """
        Node: Rank and filter tags for final output
        """
        if not state['retrieved_tags']:
            print("⚠ No tags to rank")
            state['processing_complete'] = True
            return state
        
        print("📊 Ranking tags...")
        
        # Sort by similarity score
        sorted_tags = sorted(
            state['retrieved_tags'],
            key=lambda x: x['similarity'],
            reverse=True
        )
        
        # Select top tags with confidence > threshold
        final_tags = []
        confidence_scores = {}
        
        for tag in sorted_tags:
            tag_name = tag['tag_name']
            similarity = tag['similarity']
            
            if similarity >= self.similarity_threshold:
                final_tags.append(tag_name)
                confidence_scores[tag_name] = round(similarity, 3)
        
        state['final_tags'] = final_tags
        state['confidence_scores'] = confidence_scores
        state['processing_complete'] = True
        
        print(f"  ✓ Selected {len(final_tags)} tags")
        
        return state
    
    def _should_continue(self, state: TaggingState) -> str:
        """
        Conditional edge: Check if we should continue or end
        """
        if state['vision_error']:
            return "error"
        return "continue"
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        """
        # Initialize graph
        workflow = StateGraph(TaggingState)
        
        # Add nodes
        workflow.add_node("analyze_image", self._analyze_image_node)
        workflow.add_node("retrieve_tags", self._retrieve_tags_node)
        workflow.add_node("rank_tags", self._rank_tags_node)
        
        # Add edges
        workflow.set_entry_point("analyze_image")
        
        workflow.add_conditional_edges(
            "analyze_image",
            self._should_continue,
            {
                "continue": "retrieve_tags",
                "error": END
            }
        )
        
        workflow.add_edge("retrieve_tags", "rank_tags")
        workflow.add_edge("rank_tags", END)
        
        return workflow.compile()
    
    def tag_image(self, image_path: str) -> Dict[str, Any]:
        """
        Tag a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tagging results with tags and metadata
        """
        # Initialize state
        initial_state: TaggingState = {
            'image_path': image_path,
            'image_description': None,
            'vision_error': None,
            'retrieved_tags': [],
            'final_tags': [],
            'confidence_scores': {},
            'processing_complete': False,
            'metadata': {}
        }
        
        print(f"\n{'='*60}")
        print(f"🖼️  Processing: {Path(image_path).name}")
        print('='*60)
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Format output
        result = {
            'image_path': image_path,
            'success': final_state['processing_complete'] and not final_state['vision_error'],
            'error': final_state['vision_error'],
            'description': final_state['image_description'],
            'tags': final_state['final_tags'],
            'confidence_scores': final_state['confidence_scores'],
            'metadata': final_state['metadata']
        }
        
        print(f"\n✅ Complete: {len(result['tags'])} tags assigned")
        
        return result
    
    def tag_images_batch(
        self,
        image_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Tag multiple images
        
        Args:
            image_paths: List of image paths
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of tagging results
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths, 1):
            if progress_callback:
                progress_callback(i, total)
            
            result = self.tag_image(path)
            results.append(result)
        
        return results


# Convenience function
def create_tagging_chain(
    tag_json_path: str = "51標籤庫.json",
    vision_base_url: str = "http://localhost:1234/v1",
    chroma_persist_dir: str = "./chroma_db",
    force_reload_tags: bool = False
) -> ImageTaggingChain:
    """
    Create a fully configured image tagging chain
    
    Args:
        tag_json_path: Path to tag definitions JSON
        vision_base_url: LM Studio API endpoint
        chroma_persist_dir: ChromaDB persistence directory
        force_reload_tags: Force reload tag embeddings
        
    Returns:
        Configured ImageTaggingChain
    """
    # Initialize components
    print("🚀 Initializing Image Tagging Chain...")
    print()
    
    # Initialize tag store
    print("1️⃣ Loading tag vector store...")
    tag_store = TagVectorStore(persist_directory=chroma_persist_dir)
    tag_store.initialize_from_json(tag_json_path, force_reload=force_reload_tags)
    print()
    
    # Initialize vision client
    print("2️⃣ Initializing GLM-4V client...")
    vision_client = GLM4VisionClient(base_url=vision_base_url)
    print()
    
    # Create chain
    print("3️⃣ Building LangGraph workflow...")
    chain = ImageTaggingChain(
        tag_store=tag_store,
        vision_client=vision_client
    )
    print()
    
    print("✨ Chain ready!")
    
    return chain


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tagging_chain.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create chain
    chain = create_tagging_chain()
    
    # Tag image
    result = chain.tag_image(image_path)
    
    # Display results
    print("\n" + "="*60)
    print("📋 FINAL RESULTS")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Success: {result['success']}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    if result['description']:
        print(f"\nDescription:\n{result['description'][:500]}...")
    
    if result['tags']:
        print(f"\nTags ({len(result['tags'])}):")
        for tag in result['tags'][:10]:
            score = result['confidence_scores'].get(tag, 0)
            print(f"  • {tag} ({score:.2f})")
