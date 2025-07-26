"""
Embedding Generator for Quiz Craft
Generates embeddings for text chunks and stores them in Qdrant vector database.
"""

import json
import logging
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles embedding generation and Qdrant storage for text chunks.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 qdrant_url: str = None,
                 qdrant_api_key: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            qdrant_url: Qdrant cloud instance URL
            qdrant_api_key: Qdrant API key
        """
        self.model_name = model_name
        self.model = None
        self.qdrant_client = None
        self.collection_name = "quiz_chunks"
        
        # Qdrant configuration from environment variables
        import os
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY')

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL must be provided either as parameter or environment variable")
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY must be provided either as parameter or environment variable")
        
        self._initialize_model()
        self._initialize_qdrant()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model with timing."""
        start_time = time.time()
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s - Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client with timing."""
        start_time = time.time()
        try:
            logger.info("Connecting to Qdrant...")
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )

            # Test connection
            collections = self.qdrant_client.get_collections()
            connect_time = time.time() - start_time
            logger.info(f"Connected to Qdrant in {connect_time:.2f}s - Collections: {len(collections.collections)}")

            # Create collection if it doesn't exist
            self._ensure_collection_exists()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise e
    
    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist with proper indexes."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")

                # Get embedding dimension
                embedding_dim = self.model.get_sentence_embedding_dimension()

                # Import required models for payload schema
                from qdrant_client.models import PayloadSchemaType

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )

                # Create indexes for filtering
                logger.info("Creating indexes for metadata filtering...")

                # Create keyword index for text fields
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="pdf_filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )

                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="job_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )

                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="section",
                    field_schema=PayloadSchemaType.KEYWORD
                )

                # Create integer index for numeric fields
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="page",
                    field_schema=PayloadSchemaType.INTEGER
                )

                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="token_count",
                    field_schema=PayloadSchemaType.INTEGER
                )

                logger.info(f"✅ Collection '{self.collection_name}' created with indexes")
            else:
                logger.info(f"✅ Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            raise e
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with timing."""
        start_time = time.time()
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise e
    
    def store_embeddings_batch(self,
                              chunks: List[Dict[str, Any]],
                              embeddings: np.ndarray,
                              pdf_metadata: Dict[str, Any] = None,
                              batch_size: int = 100) -> bool:
        """
        Store embeddings with rich metadata in Qdrant.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Corresponding embeddings
            pdf_metadata: PDF-level metadata (filename, job_id, etc.)
            batch_size: Number of points to upload per batch

        Returns:
            True if successful
        """
        start_time = time.time()
        try:
            logger.info(f"Storing {len(chunks)} embeddings in Qdrant...")

            # Prepare points for upload with enhanced metadata
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create comprehensive payload
                payload = {
                    # Core chunk data
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "page": chunk["page"],
                    "section": chunk.get("section"),
                    "token_count": chunk["token_count"],
                    "paragraph_index": chunk.get("paragraph_index"),
                    "chunk_order": i,

                    # PDF metadata (if provided)
                    "pdf_filename": pdf_metadata.get("filename") if pdf_metadata else "unknown.pdf",
                    "job_id": pdf_metadata.get("job_id") if pdf_metadata else None,
                    "page_range_requested": pdf_metadata.get("page_range") if pdf_metadata else None,
                    "upload_timestamp": pdf_metadata.get("timestamp") if pdf_metadata else None,

                    # Processing metadata
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                    "created_at": datetime.now().isoformat(),

                    # Original metadata
                    "original_metadata": chunk.get("metadata", {})
                }

                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique Qdrant point ID
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
            
            # Upload in batches
            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
                total_uploaded += len(batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}: {total_uploaded}/{len(points)} points")
            
            logger.info(f"✅ Successfully stored {total_uploaded} embeddings in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            raise e
    
    def process_json_file(self, json_file_path: str) -> bool:
        """
        Process a JSON file containing chunks and generate/store embeddings.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Processing JSON file: {json_file_path}")
            
            # Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            if not chunks:
                logger.warning("No chunks found in JSON file")
                return False
            
            logger.info(f"Found {len(chunks)} chunks to process")
            
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Store in Qdrant
            success = self.store_embeddings_batch(chunks, embeddings)
            
            if success:
                logger.info(f"✅ Successfully processed {len(chunks)} chunks from {json_file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to process JSON file: {e}")
            return False
    
    def search_similar_chunks(self,
                             query_text: str,
                             limit: int = 5,
                             score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score (lowered to 0.3)

        Returns:
            List of similar chunks with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query_text])[0]

            # Search in Qdrant using the newer query_points method
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results.points:
                results.append({
                    "score": result.score,
                    "chunk_id": result.payload["chunk_id"],
                    "content": result.payload["content"],
                    "page": result.payload["page"],
                    "section": result.payload.get("section"),
                    "token_count": result.payload["token_count"]
                })
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar chunks: {e}")
            return []

    def get_chunks_by_filter(self,
                            filter_conditions: Dict[str, Any] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get chunks by metadata filter (no semantic search).

        Args:
            filter_conditions: Qdrant filter conditions
            limit: Maximum number of results

        Returns:
            List of chunks with metadata
        """
        try:
            # Use scroll to get points by filter
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't need vectors for metadata queries
            )

            # Format results
            results = []
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                results.append({
                    "qdrant_id": point.id,
                    "chunk_id": point.payload["chunk_id"],
                    "content": point.payload["content"],
                    "page": point.payload["page"],
                    "section": point.payload.get("section"),
                    "token_count": point.payload["token_count"],
                    "pdf_filename": point.payload.get("pdf_filename"),
                    "job_id": point.payload.get("job_id"),
                    "page_range_requested": point.payload.get("page_range_requested"),
                    "upload_timestamp": point.payload.get("upload_timestamp"),
                    "created_at": point.payload.get("created_at")
                })

            logger.info(f"Found {len(results)} chunks matching filter")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to get chunks by filter: {e}")
            return []

    def get_chunks_by_pdf(self, pdf_filename: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific PDF."""
        filter_conditions = {
            "must": [{"key": "pdf_filename", "match": {"value": pdf_filename}}]
        }
        return self.get_chunks_by_filter(filter_conditions)

    def get_chunks_by_job_id(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific job."""
        filter_conditions = {
            "must": [{"key": "job_id", "match": {"value": job_id}}]
        }
        return self.get_chunks_by_filter(filter_conditions)

    def get_chunks_by_page_range(self, min_page: int, max_page: int) -> List[Dict[str, Any]]:
        """Get chunks within a page range."""
        filter_conditions = {
            "must": [
                {"key": "page", "range": {"gte": min_page, "lte": max_page}}
            ]
        }
        return self.get_chunks_by_filter(filter_conditions)

    def get_pdfs_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all processed PDFs.

        Returns:
            List of PDF summaries with chunk counts
        """
        try:
            # Get all points to analyze
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True,
                with_vectors=False
            )

            # Group by PDF
            pdf_stats = {}
            for point in scroll_result[0]:
                pdf_filename = point.payload.get("pdf_filename", "unknown.pdf")
                job_id = point.payload.get("job_id")
                page_range = point.payload.get("page_range_requested")
                upload_time = point.payload.get("upload_timestamp")

                if pdf_filename not in pdf_stats:
                    pdf_stats[pdf_filename] = {
                        "pdf_filename": pdf_filename,
                        "job_id": job_id,
                        "page_range_requested": page_range,
                        "upload_timestamp": upload_time,
                        "chunk_count": 0,
                        "pages": set(),
                        "total_tokens": 0
                    }

                pdf_stats[pdf_filename]["chunk_count"] += 1
                pdf_stats[pdf_filename]["pages"].add(point.payload.get("page"))
                pdf_stats[pdf_filename]["total_tokens"] += point.payload.get("token_count", 0)

            # Convert to list and format
            summaries = []
            for pdf_data in pdf_stats.values():
                pdf_data["pages_processed"] = sorted(list(pdf_data["pages"]))
                pdf_data["page_count"] = len(pdf_data["pages"])
                del pdf_data["pages"]  # Remove set for JSON serialization
                summaries.append(pdf_data)

            # Sort by upload time (newest first)
            summaries.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)

            logger.info(f"Found {len(summaries)} PDFs in Qdrant")
            return summaries

        except Exception as e:
            logger.error(f"❌ Failed to get PDFs summary: {e}")
            return []

    def search_with_filter(self,
                          query_text: str,
                          filter_conditions: Dict[str, Any] = None,
                          limit: int = 5,
                          score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Semantic search with metadata filtering.

        Args:
            query_text: Text to search for
            filter_conditions: Qdrant filter conditions
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of similar chunks with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query_text])[0]

            # Search in Qdrant with filter
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            results = []
            for result in search_results.points:
                results.append({
                    "score": result.score,
                    "qdrant_id": result.id,
                    "chunk_id": result.payload["chunk_id"],
                    "content": result.payload["content"],
                    "page": result.payload["page"],
                    "section": result.payload.get("section"),
                    "token_count": result.payload["token_count"],
                    "pdf_filename": result.payload.get("pdf_filename"),
                    "job_id": result.payload.get("job_id")
                })

            logger.info(f"Found {len(results)} similar chunks with filter")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to search with filter: {e}")
            return []


