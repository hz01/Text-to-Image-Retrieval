"""
Image Retrieval - Inference Mode
Fast image search using pre-embedded images from ChromaDB

Run 'embed_images.py' first if you haven't embedded your images yet!
"""

import os
import traceback
from pathlib import Path
from image_retrieval import ImageRetrieval
from config import (
    CHROMA_DB_PATH,
    RESULTS_FOLDER,
    MODEL_NAME,
    TOP_K,
    COLLECTION_NAME
)


def main():
    """Run image retrieval inference on pre-embedded images"""
    
    # Create results folder
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("  Text-to-Image Retrieval System")
    print("  Fast Inference Mode")
    print("=" * 60)
    print()
    
    # Check if ChromaDB exists
    if not os.path.exists(CHROMA_DB_PATH):
        print("ERROR: ChromaDB not found!")
        print()
        print("Please run 'python embed_images.py' first to embed your images.")
        input("\nPress Enter to exit...")
        return
    
    # Initialize retrieval engine (only loads the model, no image encoding)
    print("Loading CLIP model and connecting to ChromaDB...")
    retrieval = ImageRetrieval(
        model_name=MODEL_NAME,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Check if database has images
    stats = retrieval.get_stats()
    if stats['total_images'] == 0:
        print()
        print("ERROR: No images found in database!")
        print()
        print("Please run 'python embed_images.py' first to embed your images.")
        input("\nPress Enter to exit...")
        return
    
    print()
    print(f"Database Stats:")
    print(f"   - Total images indexed: {stats['total_images']}")
    print(f"   - Collection: {stats['collection_name']}")
    print(f"   - Device: {stats['device']}")
    print()
    
    print("=" * 60)
    print("  Interactive Image Search")
    print("=" * 60)
    print()
    print("TIP: Enter text queries to search for images")
    print("TIP: Type 'quit', 'exit', or 'q' to close")
    print("TIP: Type 'stats' to see database statistics")
    print("TIP: Type 'clear' to clear image cache")
    print("TIP: Results will be displayed and saved to the 'results' folder")
    print()
    
    # Interactive mode
    while True:
        print("-" * 60)
        query = input("Your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the Image Retrieval System!")
            break
        
        if query.lower() == 'stats':
            stats = retrieval.get_stats()
            print(f"\nDatabase Stats:")
            print(f"   - Total images: {stats['total_images']}")
            print(f"   - Collection: {stats['collection_name']}")
            print(f"   - Device: {stats['device']}")
            print()
            continue
        
        if query.lower() == 'clear':
            retrieval.clear_cache()
            print()
            continue
        
        if not query:
            continue
        
        try:
            print(f"\nSearching for: '{query}'...")
            
            # Generate safe filename
            safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
            safe_filename = safe_filename[:50].strip().replace(' ', '_')
            save_path = os.path.join(RESULTS_FOLDER, f"{safe_filename}.png")
            
            # Retrieve and visualize
            retrieval.visualize_results(query, top_k=TOP_K, save_path=save_path)
            print(f"Results saved to: {save_path}")
            print()
            
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

