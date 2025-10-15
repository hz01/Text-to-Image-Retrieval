"""
Image Retrieval API
Simple Flask server for text-to-image retrieval

Usage:
    python api.py
"""

import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from image_retrieval import ImageRetrieval
from config import (
    CHROMA_DB_PATH,
    MODEL_NAME,
    COLLECTION_NAME
)


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global retrieval engine
retrieval_engine = None


def initialize_retrieval_engine():
    """Initialize the retrieval engine on first request"""
    global retrieval_engine
    
    if retrieval_engine is not None:
        return
    
    if not os.path.exists(CHROMA_DB_PATH):
        print("ERROR: ChromaDB not found! Please run 'embed_images.py' first.")
        return
    
    print("Initializing Image Retrieval API...")
    retrieval_engine = ImageRetrieval(
        model_name=MODEL_NAME,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    if retrieval_engine.collection.count() == 0:
        print("WARNING: No images in database! Please run 'embed_images.py' first.")
    else:
        print(f"API ready with {retrieval_engine.collection.count()} images")


@app.route('/search', methods=['POST', 'GET'])
def search_images():
    """
    Search for images using text query
    Returns base64 encoded images with similarity scores
    
    POST Body:
        {
            "query": "text query",
            "top_k": 5
        }
    
    GET Params:
        ?query=text&top_k=5
    """
    # Initialize engine if not already done
    initialize_retrieval_engine()
    
    if retrieval_engine is None:
        return jsonify({
            'error': 'Retrieval engine not initialized. Run embed_images.py first.'
        }), 503
    
    # Get parameters from POST or GET
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
    else:  # GET
        query = request.args.get('query', '')
        top_k = int(request.args.get('top_k', 5))
    
    # Validate query
    if not query or query.strip() == '':
        return jsonify({
            'error': 'Query cannot be empty'
        }), 400
    
    # Validate top_k
    if top_k < 1 or top_k > 50:
        return jsonify({
            'error': 'top_k must be between 1 and 50'
        }), 400
    
    try:
        # Retrieve images
        images, scores = retrieval_engine.retrieve(
            query,
            top_k=top_k,
            return_scores=True
        )
        
        # Convert images to base64
        results = []
        for img, score in zip(images, scores):
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            results.append({
                "image": f"data:image/jpeg;base64,{img_base64}",
                "score": float(score)
            })
        
        return jsonify({
            "query": query,
            "results": results,
            "total_results": len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    initialize_retrieval_engine()
    
    if retrieval_engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Retrieval engine not initialized. Run embed_images.py first.'
        }), 503
    
    return jsonify({
        'status': 'online',
        'message': 'Image Retrieval API is running',
        'total_images': retrieval_engine.collection.count()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  Starting Image Retrieval API Server")
    print("=" * 60)
    print()
    
    # Initialize on startup
    initialize_retrieval_engine()
    
    print()
    print("Server running at: http://localhost:8000")
    print("Docs: Open index.html in your browser")
    print()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)
