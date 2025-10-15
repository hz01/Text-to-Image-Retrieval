# Text-to-Image Retrieval System

A semantic image search engine powered by OpenAI's CLIP (Contrastive Language-Image Pre-training) model and ChromaDB vector database. Search through your image collection using natural language queries.

## Overview

This system enables intelligent image retrieval by understanding the semantic meaning of text queries. Unlike traditional keyword-based search, it comprehends concepts and finds visually relevant images even when exact keywords don't match.

**Key Features:**
- Semantic search using natural language descriptions
- Fast vector similarity search with ChromaDB
- Multiple interfaces: CLI, REST API, and Web UI
- Persistent vector database for instant queries
- Support for multiple CLIP model variants
- Batch processing for efficient encoding

## How It Works

1. **Image Encoding**: Images are encoded into 512-dimensional vectors using CLIP's vision encoder
2. **Text Encoding**: Search queries are encoded into the same vector space using CLIP's text encoder
3. **Similarity Search**: ChromaDB performs fast nearest-neighbor search to find matching images
4. **Results**: Returns images ranked by semantic similarity to the query

## Architecture

```
┌─────────────┐
│   Images    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  CLIP Encoder   │  (Vision Model)
└────────┬────────┘
         │
         ▼
   ┌─────────────┐
   │  ChromaDB   │  (Vector Storage)
   └─────┬───────┘
         │
         ▼
┌──────────────────┐
│  Text Query      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CLIP Encoder    │  (Text Model)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Similarity Search│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Results      │
└──────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hz01/Text-to-Image-Retrieval.git
cd Text-to-Image-Retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the dataset folder:
```bash
mkdir dataset
```

4. Add your images to the `dataset` folder (supports: .jpg, .jpeg, .png, .bmp, .webp)

## Usage

### Step 1: Embed Images

Before searching, you must encode your images and store them in the vector database:

```bash
python embed_images.py
```

This process:
- Loads all images from the `dataset` folder
- Encodes them using the CLIP model
- Stores embeddings in ChromaDB at `./chroma_db`
- Only needs to be run once (or when adding new images)

### Step 2: Search Images

You have three options for searching:

#### Option A: CLI Interface (Recommended for Testing)

```bash
python main.py
```

Interactive terminal interface with visualization:
- Enter natural language queries
- View results in matplotlib window
- Results saved to `results` folder

#### Option B: Web Interface (Best User Experience)

1. Start the API server:
```bash
python api.py
```

2. Open `index.html` in your browser

Features:
- Beautiful, modern UI
- Real-time search
- Adjustable result count
- Image preview modal
- Similarity scores

#### Option C: REST API (For Integration)

Start the server:
```bash
python api.py
```

API Endpoints:

**Search Images**
```bash
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "a dog playing in the park",
  "top_k": 5
}
```

**Health Check**
```bash
GET http://localhost:8000/
```

Response format:
```json
{
  "query": "a dog playing in the park",
  "results": [
    {
      "image": "data:image/jpeg;base64,...",
      "score": 0.85
    }
  ],
  "total_results": 5
}
```

## Configuration

Edit `config.py` to customize settings:

```python
# Paths
DATASET_FOLDER = "dataset"          # Your images location
CHROMA_DB_PATH = "./chroma_db"      # Vector database path
RESULTS_FOLDER = "results"          # Search results output

# Model Configuration
MODEL_NAME = "ViT-B/32"             # CLIP model variant
# Options:
#   - "ViT-L/14"  → Best quality, slower
#   - "ViT-B/32"  → Balanced (default)
#   - "ViT-B/16"  → Good quality
#   - "RN50"      → Fastest

# Processing
BATCH_SIZE = 32                     # Reduce if out of memory
TOP_K = 5                           # Default results count

# Database
COLLECTION_NAME = "image_embeddings"
WIPE_DB_ON_EMBED = True             # Clear DB before embedding
```

## Project Structure

```
Text-to-Image-Retrieval/
├── image_retrieval.py      # Core retrieval engine class
├── config.py               # Configuration settings
├── embed_images.py         # Image embedding script
├── main.py                 # CLI interface
├── api.py                  # Flask REST API server
├── index.html              # Web UI
├── requirements.txt        # Python dependencies
├── dataset/                # Your images (create this)
├── chroma_db/              # Vector database (auto-created)
└── results/                # Search results (auto-created)
```

## Technical Details

### CLIP Model

CLIP (Contrastive Language-Image Pre-training) is a neural network trained on 400 million image-text pairs. It learns to map images and text into a shared embedding space where semantically similar concepts have similar vector representations.

**Key Capabilities:**
- Zero-shot learning: understands concepts without explicit training
- Multimodal: processes both images and text
- Semantic understanding: matches concepts, not just keywords
- Transfer learning: generalizes across domains

### ChromaDB

ChromaDB is an open-source vector database optimized for:
- Fast similarity search using HNSW (Hierarchical Navigable Small World) algorithm
- Persistent storage on disk
- Cosine similarity for normalized embeddings
- Efficient batch operations

## Troubleshooting

### Issue: "ChromaDB not found"
**Solution**: Run `python embed_images.py` first to create the database

### Issue: Out of memory during embedding
**Solution**: Reduce `BATCH_SIZE` in `config.py` (try 16 or 8)

### Issue: Poor search results
**Solutions**:
- Try a different CLIP model (ViT-L/14 for better quality)
- Ensure images are relevant to your queries
- Use more descriptive queries

### Issue: API server not starting
**Solution**: Check if port 8000 is available or change the port in `api.py`

## Dependencies

Core libraries:
- `torch` - PyTorch deep learning framework
- `clip` - OpenAI's CLIP model
- `chromadb` - Vector database
- `flask` - REST API server
- `Pillow` - Image processing
- `matplotlib` - Visualization

See `requirements.txt` for complete list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project uses OpenAI's CLIP model, which is also under the MIT License. See the [CLIP repository](https://github.com/openai/CLIP) for more information.


## Acknowledgments

- OpenAI for the CLIP model
- ChromaDB team for the vector database
- PyTorch community

---

**Built with CLIP + ChromaDB for intelligent image retrieval**

