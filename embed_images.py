"""
Embed Images Script
Run this script once to encode all images in the dataset folder and store them in ChromaDB
"""

import os
from pathlib import Path
from image_retrieval import ImageRetrieval
from config import (
    DATASET_FOLDER,
    CHROMA_DB_PATH,
    MODEL_NAME,
    BATCH_SIZE,
    COLLECTION_NAME,
    SUPPORTED_FORMATS,
    WIPE_DB_ON_EMBED
)


def main():
    """Embed all images in the dataset folder and store in ChromaDB"""
    
    print("=" * 60)
    print("  Image Embedding Script")
    print("  Encode images and store in ChromaDB Vector Database")
    print("=" * 60)
    print()
    
    # Create dataset folder if it doesn't exist
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    
    # Check if dataset folder has images
    dataset_path = Path(DATASET_FOLDER)
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(list(dataset_path.glob(f"*{ext}")))
        image_files.extend(list(dataset_path.glob(f"*{ext.upper()}")))
    
    # Remove duplicates (Windows is case-insensitive)
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"WARNING: No images found in '{DATASET_FOLDER}' folder!")
        print(f"Please add images to the '{DATASET_FOLDER}' folder and run again.")
        print()
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .webp")
        input("\nPress Enter to exit...")
        return
    
    print(f"Found {len(image_files)} images in '{DATASET_FOLDER}' folder")
    print()
    
    # Initialize retrieval engine with ChromaDB
    print("Initializing CLIP model and ChromaDB...")
    retrieval = ImageRetrieval(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print()
    
    # Wipe database if configured to do so
    if WIPE_DB_ON_EMBED and retrieval.collection.count() > 0:
        print(f"WARNING: WIPE_DB_ON_EMBED is set to True in config.py")
        print(f"Wiping existing database ({retrieval.collection.count()} images)...")
        retrieval.reset_database()
        print("Database wiped successfully!")
        print()
    
    # Ask user if they want to update existing embeddings
    if not WIPE_DB_ON_EMBED and retrieval.collection.count() > 0:
        print(f"WARNING: Database already contains {retrieval.collection.count()} images.")
        update = input("Do you want to re-embed existing images? (y/N): ").strip().lower()
        update_existing = update == 'y'
    else:
        update_existing = False
    
    print()
    
    # Load and encode images
    print("Processing images...")
    retrieval.load_images_from_directories([DATASET_FOLDER])
    retrieval.encode_and_store_images(update_existing=update_existing)
    
    # Show final stats
    stats = retrieval.get_stats()
    print()
    print("=" * 60)
    print("  Embedding Complete!")
    print("=" * 60)
    print(f"Total images in database: {stats['total_images']}")
    print(f"Database location: {CHROMA_DB_PATH}")
    print(f"Collection name: {stats['collection_name']}")
    print()
    print("TIP: You can now run 'python main.py' for fast inference!")
    print()


if __name__ == "__main__":
    main()

