"""
Configuration file for Image Retrieval System
Edit these settings to customize your setup
"""

# Paths
DATASET_FOLDER = "dataset"          # Folder containing your images
CHROMA_DB_PATH = "./chroma_db"      # ChromaDB storage location
RESULTS_FOLDER = "results"          # Where to save search results

# Model Configuration
MODEL_NAME = "ViT-B/32"             # CLIP model variant
                                    # Options: "RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"
                                    # ViT-L/14 = best quality but slower
                                    # ViT-B/32 = good balance (default)
                                    # RN50 = fastest

# Processing Settings
BATCH_SIZE = 32                     # Batch size for encoding (reduce if out of memory)
TOP_K = 5                           # Number of search results to return

# Database Settings
COLLECTION_NAME = "image_embeddings"  # ChromaDB collection name
WIPE_DB_ON_EMBED = True              # Set to True to clear database before embedding

# Supported Image Formats
SUPPORTED_FORMATS = (
    '.jpg', 
    '.jpeg', 
    '.png', 
    '.bmp', 
    '.webp'
)

