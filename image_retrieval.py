"""
ImageRetrieval Class - CLIP-based image retrieval with ChromaDB storage
"""

import os
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt
import chromadb
from chromadb.config import Settings
import hashlib


"""
What is CLIP?
=============

CLIP (Contrastive Language-Image Pre-training) is a neural network model developed by OpenAI
that learns visual concepts from natural language descriptions.

Key Features:
-------------
1. **Multimodal Learning**: CLIP was trained on 400 million (image, text) pairs from the internet,
   learning to connect visual content with text descriptions.

2. **Dual Encoders**: 
   - Image Encoder: Converts images into 512-dimensional vectors (embeddings)
   - Text Encoder: Converts text into the SAME 512-dimensional vector space
   
3. **Shared Embedding Space**: Both images and text are mapped to the same vector space,
   meaning similar concepts (whether image or text) have similar embeddings.

4. **Zero-Shot Capability**: CLIP can understand concepts it wasn't explicitly trained for,
   because it learned general visual-semantic relationships.

How It Works in This Project:
------------------------------
1. We encode all images into 512-dim vectors using CLIP's image encoder
2. We encode text queries into 512-dim vectors using CLIP's text encoder
3. We find images whose vectors are closest to the query vector (cosine similarity)
4. Result: Images that match the semantic meaning of the text!

Example:
--------
Query: "a dog playing in the park"
→ CLIP converts this to a vector [0.23, -0.45, 0.67, ...]
→ We find images with similar vectors
→ Returns: Images of dogs playing outdoors

Why CLIP is Powerful:
--------------------
- Understands CONCEPTS, not just keywords (e.g., "sunset" ≈ "orange sky at dusk")
- Works across languages and visual styles
- Pre-trained on diverse internet data, so it generalizes well
- Fast inference: encoding is just a forward pass through the network
"""


class ImageRetrieval:
    """Text-to-Image Retrieval Engine using CLIP embeddings + ChromaDB"""
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        batch_size: int = 32,
        db_path: str = "./chroma_db",
        collection_name: str = "image_embeddings"
    ):
        """
        Initialize the retrieval engine with ChromaDB
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run model on (cuda/cpu). Auto-detects if None
            batch_size: Batch size for encoding images
            db_path: Path to ChromaDB database directory
            collection_name: Name of the ChromaDB collection
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.collection_name = collection_name
        
        print(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at: {db_path}")
        # PersistentClient: stores embeddings on disk (not in-memory)
        # This allows the database to persist between runs
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        # Collections are like tables in traditional databases
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name} ({self.collection.count()} images)")
        except:
            # metadata={"hnsw:space": "cosine"} configures the HNSW (Hierarchical Navigable Small World) 
            # index to use cosine distance, which is ideal for normalized embeddings
            # Cosine distance measures angle between vectors, ignoring magnitude
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"Created new collection: {collection_name}")
        
        self.image_paths: List[str] = []
        self.cached_images: dict = {}  # Cache for loaded images
        
    def _get_image_hash(self, image_path: str) -> str:
        """Generate a unique hash for an image file"""
        # MD5 hash is used as a unique identifier for each image
        # This allows us to detect duplicate images and avoid re-encoding them
        # We hash the file content (not just the path) so renamed/moved files are still recognized
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_images_from_directories(
        self, 
        directories: List[str],
        max_images_per_dir: Optional[int] = None,
        supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ) -> None:
        """
        Load images from multiple directories
        
        Args:
            directories: List of directory paths to load images from
            max_images_per_dir: Maximum images to load per directory (None for all)
            supported_formats: Tuple of supported image extensions
        """
        self.image_paths = []
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"WARNING: Directory '{directory}' does not exist. Skipping...")
                continue
            
            image_files = []
            for ext in supported_formats:
                image_files.extend(dir_path.glob(f"*{ext}"))
                image_files.extend(dir_path.glob(f"*{ext.upper()}"))
            
            # Remove duplicates (Windows is case-insensitive)
            image_files = list(set(image_files))
            
            if max_images_per_dir:
                image_files = image_files[:max_images_per_dir]
            
            self.image_paths.extend([str(p) for p in image_files])
            print(f"Found {len(image_files)} images in {directory}")
        
        print(f"Total images found: {len(self.image_paths)}")
    
    def encode_and_store_images(self, update_existing: bool = False) -> None:
        """
        Encode all loaded images using CLIP and store in ChromaDB
        
        Args:
            update_existing: If True, re-encode images that already exist in DB
        """
        if not self.image_paths:
            raise ValueError("No images loaded. Call load_images_from_directories first.")
        
        # Get existing image IDs in the database
        # This prevents re-encoding images that are already in the database
        existing_ids = set()
        try:
            existing_data = self.collection.get()
            existing_ids = set(existing_data['ids'])
        except:
            pass
        
        images_to_process = []
        image_ids = []
        image_metadatas = []
        
        print("Checking which images need encoding...")
        for img_path in self.image_paths:
            # Generate hash-based ID for each image (content-based, not path-based)
            img_hash = self._get_image_hash(img_path)
            
            # Skip images that already exist in database (unless update_existing=True)
            if not update_existing and img_hash in existing_ids:
                continue  # Skip already encoded images
            
            images_to_process.append(img_path)
            image_ids.append(img_hash)
            # Store metadata alongside embeddings for later retrieval
            image_metadatas.append({
                "path": img_path,
                "filename": os.path.basename(img_path)
            })
        
        if not images_to_process:
            print("All images already encoded in database!")
            return
        
        print(f"Encoding {len(images_to_process)} images...")
        
        # Process images in batches to optimize GPU usage and avoid OOM errors
        # Processing all images at once would consume too much memory
        all_embeddings = []
        
        for i in tqdm(range(0, len(images_to_process), self.batch_size)):
            # Extract a batch of image paths
            batch_paths = images_to_process[i:i + self.batch_size]
            batch_images = []
            
            # Load and preprocess each image in the batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    # preprocess resizes, crops, and normalizes the image for CLIP
                    batch_images.append(self.preprocess(image))
                except Exception as e:
                    print(f"ERROR: Error loading {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack individual images into a single tensor for batch processing
            # torch.stack() combines multiple tensors along a NEW dimension (dimension 0)
            # 
            # Before stacking: list of N tensors, each shaped [3, 224, 224]
            #   - 3 = RGB color channels (Red, Green, Blue)
            #   - 224 = height in pixels (CLIP's expected input size)
            #   - 224 = width in pixels (CLIP's expected input size)
            #
            # After stacking: single tensor shaped [N, 3, 224, 224] where:
            #   - N = batch_size (number of images in this batch, e.g., 32)
            #   - 3 = color channels
            #   - 224 × 224 = image dimensions
            #
            # .to(self.device) moves the tensor to GPU (if available) or keeps it on CPU
            # This is required before passing data to the model
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Disable gradient computation (we're not training, just inferring)
            with torch.no_grad():
                # Encode images to 512-dimensional embeddings (for ViT-B/32)
                features = self.model.encode_image(batch_tensor).float()
                
                # Normalize embeddings to unit length (L2 normalization)
                # This is CRITICAL: normalized vectors allow cosine similarity = dot product
                # Formula: v_normalized = v / ||v|| where ||v|| is the L2 norm
                features = features / features.norm(dim=-1, keepdim=True)
                
                # Move embeddings back to CPU and convert to list format for ChromaDB
                all_embeddings.extend(features.cpu().numpy().tolist())
            
            # Clear GPU memory cache to prevent accumulation across batches
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Store in ChromaDB
        if all_embeddings:
            print("Storing embeddings in ChromaDB...")
            # NOTE: We slice ids and metadatas to match all_embeddings length
            # This handles cases where some images failed to load (errors during processing)
            # Ensures all three lists (ids, embeddings, metadatas) have the same length
            self.collection.add(
                ids=image_ids[:len(all_embeddings)],
                embeddings=all_embeddings,
                metadatas=image_metadatas[:len(all_embeddings)]
            )
            print(f"Successfully stored {len(all_embeddings)} image embeddings!")
            print(f"Total images in database: {self.collection.count()}")
    
    def retrieve(
        self, 
        text: str, 
        top_k: int = 5,
        return_scores: bool = False
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[float]]]:
        """
        Retrieve images matching the text query using ChromaDB
        
        Args:
            text: Text query
            top_k: Number of top results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of PIL Images (and optionally their similarity scores)
        """
        # Encode text query using CLIP's text encoder
        # tokenize() converts text string to token IDs that CLIP understands
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            # Encode text to the same embedding space as images (512-dim for ViT-B/32)
            text_features = self.model.encode_text(text_tokens).float()
            
            # CRITICAL: Normalize text embedding the same way we normalized image embeddings
            # This ensures cosine similarity can be computed as a simple dot product
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert PyTorch tensor to list format for ChromaDB API
        # [0] extracts the first (and only) embedding from the batch
        query_embedding = text_features.cpu().numpy().tolist()[0]
        
        # Query ChromaDB using vector similarity search
        # ChromaDB uses HNSW algorithm for fast approximate nearest neighbor search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())  # Cap at total images in DB
        )
        
        # Load images
        images = []
        scores = []
        
        if results['metadatas'] and len(results['metadatas']) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                img_path = metadata['path']
                try:
                    # Load image with caching to avoid redundant disk I/O
                    # Cache is especially useful when same images appear in multiple queries
                    if img_path not in self.cached_images:
                        self.cached_images[img_path] = Image.open(img_path).convert("RGB")
                    images.append(self.cached_images[img_path])
                    
                    # IMPORTANT: ChromaDB returns DISTANCES, not similarity scores
                    # Cosine distance = 1 - cosine similarity
                    # Therefore: similarity = 1 - distance
                    # Range: distance [0, 2] → similarity [-1, 1] where 1 = identical, -1 = opposite
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    scores.append(similarity)
                except Exception as e:
                    print(f"ERROR: Error loading {img_path}: {e}")
        
        if return_scores:
            return images, scores
        return images
    
    def visualize_results(
        self, 
        text: str, 
        top_k: int = 5,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize retrieval results
        
        Args:
            text: Text query
            top_k: Number of results to display
            save_path: Optional path to save the visualization
        """
        results, scores = self.retrieve(text, top_k=top_k, return_scores=True)
        
        if not results:
            print("No results found!")
            return
        
        cols = min(5, len(results))
        rows = (len(results) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if len(results) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (img, score) in enumerate(zip(results, scores)):
            axes[i].imshow(img)
            axes[i].set_title(f"Score: {score:.3f}", fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Query: "{text}"', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        return {
            "total_images": self.collection.count(),
            "collection_name": self.collection_name,
            "device": self.device
        }
    
    def clear_cache(self) -> None:
        """Clear the image cache to free memory"""
        self.cached_images.clear()
        print("Image cache cleared")
    
    def reset_database(self) -> None:
        """Delete all data from the collection"""
        # Remove the entire collection (including all embeddings and metadata)
        self.chroma_client.delete_collection(self.collection_name)
        # Recreate it with the same configuration (cosine distance)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Must match original configuration
        )
        print("Database reset complete")

