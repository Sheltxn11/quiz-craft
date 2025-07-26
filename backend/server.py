from flask import Flask, request, jsonify
from flask_cors import CORS
import logging, os, uuid, tempfile, time
from azure.storage.blob import BlobServiceClient
from werkzeug.utils import secure_filename
import requests
from mistralai import Mistral
from dotenv import load_dotenv
from text_processor import TextProcessor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from mcq_generator import MCQGenerator
import asyncio

# Suppress TensorFlow warnings to prevent server restarts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
CORS(app, resources={r"/*": {"origins": "http://localhost:8081"}})

# Configuration from environment variables
AZURE_CONN_STR = os.getenv('AZURE_CONN_STR')
CONTAINER_NAME = os.getenv('CONTAINER_NAME', 'pdfs')
BLOB_BASE_URL = os.getenv('BLOB_BASE_URL')

# Validate required environment variables
if not AZURE_CONN_STR:
    app.logger.error("AZURE_CONN_STR not found in environment variables")
if not BLOB_BASE_URL:
    app.logger.error("BLOB_BASE_URL not found in environment variables")

# Mistral API configuration
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    app.logger.warning("MISTRAL_API_KEY not found in environment variables")

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

# Initialize text processor and chunker
text_processor = TextProcessor()
text_chunker = TextChunker(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=400,
    overlap_percentage=0.1
)

# Initialize components (lazy loading to avoid startup delays)
embedding_generator = None
mcq_generator = None

def get_embedding_generator():
    """Get or initialize the embedding generator (lazy loading)"""
    global embedding_generator
    if embedding_generator is None:
        app.logger.info("Initializing Qdrant-based embedding generator...")
        embedding_generator = EmbeddingGenerator()
        app.logger.info("✅ Qdrant embedding generator initialized")
    return embedding_generator

def get_mcq_generator():
    """Get or initialize the MCQ generator (lazy loading)"""
    global mcq_generator
    if mcq_generator is None:
        app.logger.info("Initializing MCQ generator...")
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        mcq_generator = MCQGenerator(gemini_api_key=gemini_api_key)
        app.logger.info("✅ MCQ generator initialized")
    return mcq_generator

# In-memory storage for extracted text and chunks (in production, use a proper database)
extracted_texts = {}
text_chunks = {}

def retrieve_pdf_from_blob(blob_url):
    """
    Retrieve PDF content from Azure Blob Storage using the blob service client
    """
    try:
        # Extract filename from blob URL
        filename = blob_url.split('/')[-1]
        app.logger.info(f"Retrieving blob: {filename}")

        # Use Azure Blob Service Client for authenticated access
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)

        # Download blob content
        blob_data = blob_client.download_blob()
        pdf_content = blob_data.readall()

        app.logger.info(f"Successfully retrieved PDF: {filename}, size: {len(pdf_content)} bytes")
        return pdf_content

    except Exception as e:
        app.logger.error(f"Failed to retrieve PDF from blob storage: {str(e)}")
        # Fallback to direct HTTP request (in case blob is publicly accessible)
        try:
            app.logger.info("Attempting direct HTTP request as fallback")
            response = requests.get(blob_url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as http_error:
            app.logger.error(f"HTTP fallback also failed: {str(http_error)}")
            raise Exception(f"Failed to retrieve PDF: {str(e)}")

def _extract_pdf_pages(pdf_path, page_from=None, page_to=None):
    """
    Extract specific pages from PDF and return path to new PDF with only those pages
    This is the KEY optimization - we extract pages BEFORE uploading to blob!
    """
    import PyPDF2
    import io

    try:
        # Read the original PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)

            app.logger.info(f"Original PDF has {total_pages} pages")

            # Set default values
            if page_from is None:
                page_from = 1
            if page_to is None:
                page_to = total_pages

            # Validate page range
            if page_from < 1:
                page_from = 1
            if page_to > total_pages:
                page_to = total_pages
            if page_from > page_to:
                page_from, page_to = page_to, page_from

            app.logger.info(f"Extracting pages {page_from} to {page_to} from {total_pages} total pages")

            # Create new PDF with only requested pages
            pdf_writer = PyPDF2.PdfWriter()

            # Add requested pages (convert to 0-based indexing)
            pages_added = []
            for page_num in range(page_from - 1, page_to):
                if page_num < len(pdf_reader.pages):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                    pages_added.append(page_num + 1)  # Convert back to 1-based for logging

            app.logger.info(f"Added {len(pages_added)} pages to extracted PDF: {pages_added}")

            # Write to new temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='_extracted.pdf') as extracted_tmp:
                pdf_writer.write(extracted_tmp)
                extracted_path = extracted_tmp.name

            app.logger.info(f"Created extracted PDF: {extracted_path}")
            return extracted_path

    except Exception as e:
        app.logger.error(f"Error extracting pages from PDF: {e}")
        raise e

def _save_json_output(job_id, job_data):
    """Auto-save JSON output for verification"""
    try:
        import json
        from datetime import datetime

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{job_id[:8]}_{timestamp}.json"

        # Prepare export data
        export_data = {
            'job_id': job_id,
            'timestamp': job_data['timestamp'],
            'page_range_requested': job_data.get('page_range_requested', 'all'),
            'extraction_metadata': job_data.get('extraction_metadata', {}),
            'processing_summary': {
                'total_segments': len(job_data.get('structured_data', {}).get('segments', [])),
                'total_chunks': len(job_data.get('chunks', [])),
                'pages_processed': job_data.get('metadata', {}).get('pages_processed', []),
                'chunk_statistics': job_data.get('chunk_statistics', {})
            },
            'segments': job_data.get('structured_data', {}).get('segments', []),
            'chunks': job_data.get('chunks', []),
            'metadata': {
                'text_metadata': job_data.get('metadata', {}),
                'chunk_statistics': job_data.get('chunk_statistics', {}),
                'extraction_metadata': job_data.get('extraction_metadata', {})
            }
        }

        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        app.logger.info(f"✅ Auto-saved JSON output: {filename}")

    except Exception as e:
        app.logger.error(f"❌ Failed to auto-save JSON: {e}")

def extract_text_with_mistral_ocr(blob_url, job_id, page_from=None, page_to=None):
    """
    Extract text from PDF using Mistral's Document OCR API and process it
    """
    if not mistral_client:
        raise Exception("Mistral API key not configured")

    try:
        app.logger.info(f"Starting OCR processing for job {job_id} with blob URL: {blob_url}")

        # Download PDF from blob storage
        app.logger.info(f"Downloading PDF from blob storage for job {job_id}")
        pdf_content = retrieve_pdf_from_blob(blob_url)

        # Note: PDF page extraction would be here, but Mistral OCR processes entire PDF
        # The filtering happens in text processing stage instead
        extraction_metadata = {
            'extraction_performed': False,
            'note': 'Mistral OCR processes entire PDF, filtering happens in text processing',
            'requested_pages': f"{page_from}-{page_to}" if page_from or page_to else "all"
        }

        # Convert PDF to base64
        import base64
        base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
        app.logger.info(f"Prepared PDF for OCR - job {job_id}, size: {len(base64_pdf)} characters")

        # Call Mistral OCR API with base64 encoded PDF
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            include_image_base64=False  # Set to True if you need image data
        )

        # Extract text content from response
        extracted_text = ""
        app.logger.info(f"OCR response type: {type(ocr_response)}")
        app.logger.info(f"OCR response attributes: {dir(ocr_response)}")

        # Handle different response structures from Mistral OCR
        if hasattr(ocr_response, 'content') and ocr_response.content:
            extracted_text = ocr_response.content
            app.logger.info("Using ocr_response.content")
        elif hasattr(ocr_response, 'text') and ocr_response.text:
            extracted_text = ocr_response.text
            app.logger.info("Using ocr_response.text")
        elif hasattr(ocr_response, 'pages') and ocr_response.pages:
            # Extract text from pages if response has pages structure
            app.logger.info(f"Found {len(ocr_response.pages)} pages in OCR response")

            # Create a structured format that preserves page boundaries
            page_texts = []
            for i, page in enumerate(ocr_response.pages):
                page_content = ""
                if hasattr(page, 'markdown') and page.markdown:
                    page_content = page.markdown
                    app.logger.info(f"Added markdown from page {i}, length: {len(page.markdown)}")
                elif hasattr(page, 'text') and page.text:
                    page_content = page.text
                    app.logger.info(f"Added text from page {i}, length: {len(page.text)}")

                if page_content.strip():
                    page_texts.append(f"PAGE_{i+1}_START\n{page_content}\nPAGE_{i+1}_END")

            extracted_text = "\n\n".join(page_texts)
        else:
            # Handle different response structures
            app.logger.warning(f"Unknown OCR response structure: {dir(ocr_response)}")
            extracted_text = str(ocr_response)

        # Log the structure of the extracted text for debugging
        app.logger.info(f"Extracted text preview (first 500 chars): {extracted_text[:500]}")
        if 'pages=' in extracted_text:
            app.logger.info("Detected 'pages=' in extracted text")
        if 'OCRPageObject' in extracted_text:
            app.logger.info("Detected 'OCRPageObject' in extracted text")

        app.logger.info(f"OCR processing completed for job {job_id}. Extracted {len(extracted_text)} characters")

        # Phase 3: Process and structure the text
        app.logger.info(f"Starting text processing for job {job_id}")
        try:
            structured_data = text_processor.process_text(
                raw_text=extracted_text,
                page_from=page_from,
                page_to=page_to
            )
            app.logger.info(f"Text processing completed for job {job_id}. Created {len(structured_data['segments'])} segments")

            # Phase 4: Chunk the structured text
            app.logger.info(f"Starting text chunking for job {job_id}")
            chunks = text_chunker.chunk_structured_text(structured_data)
            chunk_stats = text_chunker.get_chunk_statistics(chunks)
            app.logger.info(f"Text chunking completed for job {job_id}. Created {len(chunks)} chunks")

            # Store chunks separately for easy access
            text_chunks[job_id] = {
                'chunks': chunks,
                'statistics': chunk_stats,
                'timestamp': time.time(),
                'status': 'completed'
            }

            # Store both raw and structured text with chunking info
            extracted_texts[job_id] = {
                'raw_text': extracted_text,
                'structured_data': structured_data,
                'processed_text': structured_data['processed_text'],
                'chunks': [
                    {
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content,
                        'token_count': chunk.token_count,
                        'page': chunk.page,
                        'section': chunk.section,
                        'paragraph_index': chunk.paragraph_index,
                        'metadata': chunk.metadata
                    }
                    for chunk in chunks
                ],
                'chunk_statistics': chunk_stats,
                'metadata': structured_data['metadata'],
                'extraction_metadata': extraction_metadata,
                'page_range_requested': f"{page_from}-{page_to}" if page_from or page_to else "all",
                'timestamp': time.time(),
                'status': 'completed'
            }

            # Auto-save JSON output for verification
            _save_json_output(job_id, extracted_texts[job_id])

            # Auto-generate embeddings and store in Qdrant with rich metadata
            try:
                app.logger.info(f"Auto-generating embeddings and storing in Qdrant for job {job_id}")

                # Initialize Qdrant service
                generator = get_embedding_generator()

                # Prepare PDF metadata for Qdrant storage
                pdf_metadata = {
                    "filename": request.files.get('pdf').filename if request.files.get('pdf') else f"job_{job_id[:8]}.pdf",
                    "job_id": job_id,
                    "page_range": f"{page_from}-{page_to}" if page_from and page_to else "all",
                    "timestamp": extracted_texts[job_id]['timestamp'],
                    "extraction_metadata": extraction_metadata
                }

                # Extract text content from chunks (TextChunk objects)
                texts = [chunk.content for chunk in chunks]

                # Convert TextChunk objects to dictionaries for embedding storage
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dict = {
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content,
                        'page': chunk.page,
                        'section': chunk.section,
                        'token_count': chunk.token_count,
                        'paragraph_index': chunk.paragraph_index,
                        'metadata': chunk.metadata or {}
                    }
                    chunk_dicts.append(chunk_dict)

                # Generate embeddings
                embeddings = generator.generate_embeddings(texts)

                # Store in Qdrant with rich metadata (no PostgreSQL needed!)
                embedding_success = generator.store_embeddings_batch(
                    chunks=chunk_dicts,
                    embeddings=embeddings,
                    pdf_metadata=pdf_metadata
                )

                if embedding_success:
                    extracted_texts[job_id]['embeddings_generated'] = True
                    extracted_texts[job_id]['embeddings_count'] = len(chunks)
                    extracted_texts[job_id]['qdrant_storage'] = True
                    app.logger.info(f"✅ Auto-generated and stored {len(chunks)} embeddings in Qdrant for job {job_id}")
                else:
                    app.logger.warning(f"⚠️ Failed to store embeddings in Qdrant for job {job_id}")

            except Exception as processing_error:
                app.logger.error(f"❌ Failed to auto-generate embeddings for job {job_id}: {processing_error}")
                # Don't fail the main process if embedding generation fails

            # Convert chunks to dictionaries for JSON serialization
            chunks_for_response = []
            for chunk in chunks:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'page': chunk.page,
                    'section': chunk.section,
                    'token_count': chunk.token_count,
                    'paragraph_index': chunk.paragraph_index,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'overlap_with_previous': chunk.overlap_with_previous,
                    'overlap_with_next': chunk.overlap_with_next,
                    'metadata': chunk.metadata or {}
                }
                chunks_for_response.append(chunk_dict)

            return {
                'structured_data': structured_data,
                'chunks': chunks_for_response,
                'chunk_statistics': chunk_stats,
                'extraction_metadata': extraction_metadata,
                'embeddings_generated': extracted_texts[job_id].get('embeddings_generated', False),
                'embeddings_count': extracted_texts[job_id].get('embeddings_count', 0),
                'qdrant_storage': extracted_texts[job_id].get('qdrant_storage', False)
            }

        except Exception as processing_error:
            app.logger.error(f"Text processing failed for job {job_id}: {str(processing_error)}")
            # Fallback to raw text if processing fails
            extracted_texts[job_id] = {
                'raw_text': extracted_text,
                'processed_text': extracted_text,
                'timestamp': time.time(),
                'status': 'completed',
                'processing_error': str(processing_error)
            }
            return {'processed_text': extracted_text, 'status': 'completed'}

    except Exception as e:
        app.logger.error(f"OCR processing failed for job {job_id}: {str(e)}")
        extracted_texts[job_id] = {
            'text': None,
            'timestamp': time.time(),
            'status': 'failed',
            'error': str(e)
        }
        raise Exception(f"OCR processing failed: {str(e)}")

@app.route('/generate-mcq', methods=['POST'])
def generate_mcq():
    pdf_file = request.files.get('pdf')
    page_from = request.form.get('from')
    page_to = request.form.get('to')
    num_questions = request.form.get('num_questions')

    if not pdf_file:
        return {'error': 'No PDF file provided'}, 400

    filename = secure_filename(pdf_file.filename)
    job_id = str(uuid.uuid4())  # Generate job ID

    # Convert page parameters to integers
    try:
        page_from_int = int(page_from) if page_from else None
        page_to_int = int(page_to) if page_to else None
    except (ValueError, TypeError):
        page_from_int = None
        page_to_int = None

    # Save original PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_file.save(tmp)
        original_tmp_path = tmp.name

    # Extract requested pages BEFORE uploading to blob
    extracted_tmp_path = None
    try:
        if page_from_int is not None or page_to_int is not None:
            app.logger.info(f"Extracting pages {page_from_int}-{page_to_int} from PDF before upload")
            extracted_tmp_path = _extract_pdf_pages(original_tmp_path, page_from_int, page_to_int)
            upload_path = extracted_tmp_path
            app.logger.info(f"Successfully extracted pages, will upload clipped PDF")
        else:
            app.logger.info("No page range specified, uploading entire PDF")
            upload_path = original_tmp_path
    except Exception as extract_error:
        app.logger.error(f"Failed to extract pages: {extract_error}")
        # Fallback to original PDF if extraction fails
        upload_path = original_tmp_path

    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        # Upload the extracted/clipped PDF (not the original!)
        with open(upload_path, 'rb') as pdf_data:
            container_client.upload_blob(
                name=filename,
                data=pdf_data,
                overwrite=True,
                timeout=60  # 60 second timeout
            )

        blob_url = BLOB_BASE_URL + filename
        app.logger.info(f"Uploaded {'clipped' if extracted_tmp_path else 'original'} PDF to Azure Blob Storage: {blob_url}")
    except Exception as upload_error:
        app.logger.error(f"Failed to upload PDF to Azure: {upload_error}")
        return {'error': f'Failed to upload PDF: {str(upload_error)}'}, 500
    finally:
        # Clean up temp files
        try:
            os.remove(original_tmp_path)
            if extracted_tmp_path and extracted_tmp_path != original_tmp_path:
                os.remove(extracted_tmp_path)
        except Exception as cleanup_error:
            app.logger.warning(f"Failed to clean up temp files: {cleanup_error}")

    # Initialize job status
    extracted_texts[job_id] = {
        'text': None,
        'timestamp': time.time(),
        'status': 'processing'
    }

    # Parse page parameters
    page_from_int = None
    page_to_int = None
    try:
        if page_from:
            page_from_int = int(page_from)
        if page_to:
            page_to_int = int(page_to)
    except ValueError:
        return {'error': 'Invalid page numbers provided'}, 400

    # Start OCR processing in background (in production, use a task queue like Celery)
    try:
        extract_text_with_mistral_ocr(blob_url, job_id, page_from_int, page_to_int)
    except Exception as e:
        app.logger.error(f"Failed to start OCR processing: {str(e)}")
        return {'error': f'Failed to start OCR processing: {str(e)}'}, 500

    # Return job_id for tracking
    return {'status': 'received', 'job_id': job_id, 'blob_url': blob_url}, 200

@app.route('/ocr-status/<job_id>', methods=['GET'])
def get_ocr_status(job_id):
    """
    Check the status of OCR processing for a given job ID
    """
    if job_id not in extracted_texts:
        return {'error': 'Job ID not found'}, 404

    job_data = extracted_texts[job_id]

    # Clean up old jobs (older than 1 hour)
    current_time = time.time()
    if current_time - job_data['timestamp'] > 3600:  # 1 hour
        del extracted_texts[job_id]
        return {'error': 'Job expired'}, 410

    response = {
        'job_id': job_id,
        'status': job_data['status'],
        'timestamp': job_data['timestamp']
    }

    if job_data['status'] == 'failed':
        response['error'] = job_data.get('error', 'Unknown error')
    elif job_data['status'] == 'completed':
        # Handle both old and new data structures
        if 'processed_text' in job_data:
            response['text_length'] = len(job_data['processed_text'])
            if 'metadata' in job_data:
                response['metadata'] = job_data['metadata']
            if 'chunk_statistics' in job_data:
                response['chunk_statistics'] = job_data['chunk_statistics']
        elif 'text' in job_data:
            response['text_length'] = len(job_data['text']) if job_data['text'] else 0

    return jsonify(response), 200

@app.route('/extracted-text/<job_id>', methods=['GET'])
def get_extracted_text(job_id):
    """
    Retrieve the extracted text for a given job ID
    """
    if job_id not in extracted_texts:
        return {'error': 'Job ID not found'}, 404

    job_data = extracted_texts[job_id]

    # Clean up old jobs (older than 1 hour)
    current_time = time.time()
    if current_time - job_data['timestamp'] > 3600:  # 1 hour
        del extracted_texts[job_id]
        return {'error': 'Job expired'}, 410

    if job_data['status'] == 'processing':
        return {'error': 'OCR processing still in progress'}, 202
    elif job_data['status'] == 'failed':
        return {'error': job_data.get('error', 'OCR processing failed')}, 500
    elif job_data['status'] == 'completed':
        # Return structured data if available, otherwise raw text
        response_data = {
            'job_id': job_id,
            'timestamp': job_data['timestamp']
        }

        if 'structured_data' in job_data:
            response_data.update({
                'structured_data': job_data['structured_data'],
                'processed_text': job_data['processed_text'],
                'raw_text': job_data.get('raw_text', ''),
                'metadata': job_data.get('metadata', {})
            })
        elif 'text' in job_data:
            response_data['text'] = job_data['text']

        return response_data, 200
    else:
        return {'error': 'Unknown job status'}, 500

@app.route('/structured-text/<job_id>', methods=['GET'])
def get_structured_text(job_id):
    """
    Retrieve the structured text data for a given job ID
    """
    if job_id not in extracted_texts:
        return {'error': 'Job ID not found'}, 404

    job_data = extracted_texts[job_id]

    # Clean up old jobs (older than 1 hour)
    current_time = time.time()
    if current_time - job_data['timestamp'] > 3600:  # 1 hour
        del extracted_texts[job_id]
        if job_id in text_chunks:
            del text_chunks[job_id]
        return {'error': 'Job expired'}, 410

    if job_data['status'] == 'processing':
        return {'error': 'Text processing still in progress'}, 202
    elif job_data['status'] == 'failed':
        return {'error': job_data.get('error', 'Text processing failed')}, 500
    elif job_data['status'] == 'completed':
        if 'structured_data' in job_data:
            return {
                'job_id': job_id,
                'structured_data': job_data['structured_data'],
                'timestamp': job_data['timestamp']
            }, 200
        else:
            return {'error': 'Structured data not available'}, 404
    else:
        return {'error': 'Unknown job status'}, 500

@app.route('/chunks/<job_id>', methods=['GET'])
def get_text_chunks(job_id):
    """
    Retrieve the text chunks for a given job ID
    """
    if job_id not in text_chunks:
        return {'error': 'Job ID not found or chunks not available'}, 404

    chunk_data = text_chunks[job_id]

    # Clean up old jobs (older than 1 hour)
    current_time = time.time()
    if current_time - chunk_data['timestamp'] > 3600:  # 1 hour
        del text_chunks[job_id]
        return {'error': 'Job expired'}, 410

    if chunk_data['status'] == 'completed':
        # Convert chunks to serializable format
        chunks_json = [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'token_count': chunk.token_count,
                'page': chunk.page,
                'section': chunk.section,
                'paragraph_index': chunk.paragraph_index,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'overlap_with_previous': chunk.overlap_with_previous,
                'overlap_with_next': chunk.overlap_with_next,
                'metadata': chunk.metadata
            }
            for chunk in chunk_data['chunks']
        ]

        return {
            'job_id': job_id,
            'chunks': chunks_json,
            'statistics': chunk_data['statistics'],
            'timestamp': chunk_data['timestamp']
        }, 200
    else:
        return {'error': 'Chunks not ready'}, 202

@app.route('/chunk/<job_id>/<chunk_id>', methods=['GET'])
def get_single_chunk(job_id, chunk_id):
    """
    Retrieve a specific chunk by job_id and chunk_id
    """
    if job_id not in text_chunks:
        return {'error': 'Job ID not found'}, 404

    chunk_data = text_chunks[job_id]

    # Find the specific chunk
    target_chunk = None
    for chunk in chunk_data['chunks']:
        if chunk.chunk_id == chunk_id:
            target_chunk = chunk
            break

    if not target_chunk:
        return {'error': 'Chunk ID not found'}, 404

    return {
        'job_id': job_id,
        'chunk': {
            'chunk_id': target_chunk.chunk_id,
            'content': target_chunk.content,
            'token_count': target_chunk.token_count,
            'page': target_chunk.page,
            'section': target_chunk.section,
            'paragraph_index': target_chunk.paragraph_index,
            'start_char': target_chunk.start_char,
            'end_char': target_chunk.end_char,
            'overlap_with_previous': target_chunk.overlap_with_previous,
            'overlap_with_next': target_chunk.overlap_with_next,
            'metadata': target_chunk.metadata
        }
    }, 200

@app.route('/export/<job_id>', methods=['GET'])
def export_job_data(job_id):
    """
    Export all job data (structured text, chunks, metadata) as JSON for verification
    """
    if job_id not in extracted_texts:
        return {'error': 'Job ID not found'}, 404

    job_data = extracted_texts[job_id]

    # Clean up old jobs (older than 1 hour)
    current_time = time.time()
    if current_time - job_data['timestamp'] > 3600:  # 1 hour
        del extracted_texts[job_id]
        if job_id in text_chunks:
            del text_chunks[job_id]
        return {'error': 'Job expired'}, 410

    if job_data['status'] != 'completed':
        return {'error': 'Job not completed yet'}, 202

    # Prepare comprehensive export data
    export_data = {
        'job_id': job_id,
        'timestamp': job_data['timestamp'],
        'processing_info': {
            'pages_requested': f"Pages {request.args.get('page_from', 'all')} to {request.args.get('page_to', 'all')}",
            'total_segments': len(job_data.get('structured_data', {}).get('segments', [])),
            'total_chunks': len(job_data.get('chunks', [])),
            'raw_text_length': len(job_data.get('raw_text', '')),
            'processed_text_length': len(job_data.get('processed_text', ''))
        },
        'structured_data': job_data.get('structured_data', {}),
        'chunks': job_data.get('chunks', []),
        'chunk_statistics': job_data.get('chunk_statistics', {}),
        'metadata': job_data.get('metadata', {})
    }

    return jsonify(export_data), 200

@app.route('/export/<job_id>/file', methods=['GET'])
def export_job_data_file(job_id):
    """
    Export job data as downloadable JSON file
    """
    if job_id not in extracted_texts:
        return {'error': 'Job ID not found'}, 404

    job_data = extracted_texts[job_id]

    if job_data['status'] != 'completed':
        return {'error': 'Job not completed yet'}, 202

    # Create export data
    export_data = {
        'job_id': job_id,
        'timestamp': job_data['timestamp'],
        'processing_summary': {
            'total_segments': len(job_data.get('structured_data', {}).get('segments', [])),
            'total_chunks': len(job_data.get('chunks', [])),
            'pages_processed': job_data.get('metadata', {}).get('pages_processed', []),
            'chunk_statistics': job_data.get('chunk_statistics', {})
        },
        'segments': job_data.get('structured_data', {}).get('segments', []),
        'chunks': job_data.get('chunks', []),
        'full_metadata': {
            'text_metadata': job_data.get('metadata', {}),
            'chunk_statistics': job_data.get('chunk_statistics', {})
        }
    }

    # Convert to JSON string
    import json
    json_content = json.dumps(export_data, indent=2, ensure_ascii=False)

    # Create response with file download
    from flask import Response
    response = Response(
        json_content,
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=quiz_craft_export_{job_id[:8]}.json'
        }
    )

    return response

@app.route('/process-ocr', methods=['POST'])
def process_ocr_directly():
    """
    Direct OCR processing endpoint that takes a blob URL and returns extracted text
    """
    data = request.get_json()

    if not data or 'blob_url' not in data:
        return {'error': 'blob_url is required'}, 400

    blob_url = data['blob_url']
    page_from = data.get('page_from')
    page_to = data.get('page_to')
    job_id = str(uuid.uuid4())

    try:
        result = extract_text_with_mistral_ocr(blob_url, job_id, page_from, page_to)
        return {
            'job_id': job_id,
            'result': result,
            'status': 'completed'
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings for chunks and store in Qdrant"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')

        if not job_id:
            return {'error': 'job_id is required'}, 400

        # Check if job exists
        if job_id not in extracted_texts:
            return {'error': 'Job not found'}, 404

        job_data = extracted_texts[job_id]
        chunks = job_data.get('chunks', [])

        if not chunks:
            return {'error': 'No chunks found for this job'}, 400

        # Get embedding generator
        generator = get_embedding_generator()

        # Generate embeddings for chunks
        app.logger.info(f"Generating embeddings for {len(chunks)} chunks from job {job_id}")

        # Extract text content
        texts = [chunk['content'] for chunk in chunks]

        # Generate embeddings
        embeddings = generator.generate_embeddings(texts)

        # Store in Qdrant
        success = generator.store_embeddings_batch(chunks, embeddings)

        if success:
            # Update job data
            job_data['embeddings_generated'] = True
            job_data['embeddings_count'] = len(chunks)

            return {
                'job_id': job_id,
                'message': f'Successfully generated and stored {len(chunks)} embeddings',
                'embeddings_count': len(chunks),
                'status': 'completed'
            }, 200
        else:
            return {'error': 'Failed to store embeddings'}, 500

    except Exception as e:
        app.logger.error(f"Error generating embeddings: {e}")
        return {'error': str(e)}, 500

@app.route('/search-chunks', methods=['POST'])
def search_chunks():
    """Search for similar chunks using semantic similarity with optional filtering"""
    try:
        data = request.get_json()
        query = data.get('query')
        limit = data.get('limit', 5)
        score_threshold = data.get('score_threshold', 0.3)

        # Optional filters
        pdf_filename = data.get('pdf_filename')
        job_id = data.get('job_id')
        min_page = data.get('min_page')
        max_page = data.get('max_page')

        if not query:
            return {'error': 'query is required'}, 400

        # Get embedding generator
        generator = get_embedding_generator()

        # Build filter conditions
        filter_conditions = None
        if pdf_filename or job_id or min_page or max_page:
            must_conditions = []

            if pdf_filename:
                must_conditions.append({"key": "pdf_filename", "match": {"value": pdf_filename}})

            if job_id:
                must_conditions.append({"key": "job_id", "match": {"value": job_id}})

            if min_page and max_page:
                must_conditions.append({"key": "page", "range": {"gte": min_page, "lte": max_page}})
            elif min_page:
                must_conditions.append({"key": "page", "range": {"gte": min_page}})
            elif max_page:
                must_conditions.append({"key": "page", "range": {"lte": max_page}})

            if must_conditions:
                filter_conditions = {"must": must_conditions}

        # Search with or without filter
        if filter_conditions:
            results = generator.search_with_filter(
                query_text=query,
                filter_conditions=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
        else:
            results = generator.search_similar_chunks(
                query_text=query,
                limit=limit,
                score_threshold=score_threshold
            )

        return {
            'query': query,
            'filters_applied': filter_conditions is not None,
            'filter_conditions': filter_conditions,
            'results_count': len(results),
            'results': results,
            'source': 'qdrant'
        }, 200

    except Exception as e:
        app.logger.error(f"Error searching chunks: {e}")
        return {'error': str(e)}, 500

@app.route('/embedding-status', methods=['GET'])
def embedding_status():
    """Get status of embedding generation for all jobs"""
    try:
        status = {}
        for job_id, job_data in extracted_texts.items():
            status[job_id] = {
                'has_chunks': len(job_data.get('chunks', [])) > 0,
                'chunk_count': len(job_data.get('chunks', [])),
                'embeddings_generated': job_data.get('embeddings_generated', False),
                'embeddings_count': job_data.get('embeddings_count', 0),
                'timestamp': job_data.get('timestamp'),
                'page_range': job_data.get('page_range_requested', 'unknown')
            }

        return {
            'jobs': status,
            'total_jobs': len(status)
        }, 200

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/qdrant/pdfs', methods=['GET'])
def get_pdfs_from_qdrant():
    """Get metadata for all processed PDFs from Qdrant"""
    try:
        generator = get_embedding_generator()
        pdfs = generator.get_pdfs_summary()

        return {
            'pdfs': pdfs,
            'total_count': len(pdfs),
            'source': 'qdrant'
        }, 200

    except Exception as e:
        app.logger.error(f"Error retrieving PDFs from Qdrant: {e}")
        return {'error': str(e)}, 500

@app.route('/qdrant/pdf/<pdf_filename>/chunks', methods=['GET'])
def get_pdf_chunks_from_qdrant(pdf_filename):
    """Get all chunks for a specific PDF from Qdrant"""
    try:
        generator = get_embedding_generator()
        chunks = generator.get_chunks_by_pdf(pdf_filename)

        return {
            'pdf_filename': pdf_filename,
            'chunks': chunks,
            'total_count': len(chunks),
            'source': 'qdrant'
        }, 200

    except Exception as e:
        app.logger.error(f"Error retrieving chunks for PDF {pdf_filename} from Qdrant: {e}")
        return {'error': str(e)}, 500

@app.route('/qdrant/job/<job_id>/chunks', methods=['GET'])
def get_job_chunks_from_qdrant(job_id):
    """Get all chunks for a specific job from Qdrant"""
    try:
        generator = get_embedding_generator()
        chunks = generator.get_chunks_by_job_id(job_id)

        return {
            'job_id': job_id,
            'chunks': chunks,
            'total_count': len(chunks),
            'source': 'qdrant'
        }, 200

    except Exception as e:
        app.logger.error(f"Error retrieving chunks for job {job_id} from Qdrant: {e}")
        return {'error': str(e)}, 500

@app.route('/qdrant/pages/<int:min_page>/<int:max_page>/chunks', methods=['GET'])
def get_page_range_chunks_from_qdrant(min_page, max_page):
    """Get chunks within a page range from Qdrant"""
    try:
        generator = get_embedding_generator()
        chunks = generator.get_chunks_by_page_range(min_page, max_page)

        return {
            'page_range': f"{min_page}-{max_page}",
            'chunks': chunks,
            'total_count': len(chunks),
            'source': 'qdrant'
        }, 200

    except Exception as e:
        app.logger.error(f"Error retrieving chunks for pages {min_page}-{max_page} from Qdrant: {e}")
        return {'error': str(e)}, 500

@app.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    """Generate MCQs from search query using the complete pipeline"""
    try:
        data = request.get_json()

        # Required parameters
        query = data.get('query')
        if not query:
            return {'error': 'query is required'}, 400

        # Optional parameters with defaults
        num_questions = data.get('num_questions', 5)
        difficulty = data.get('difficulty', 'medium')
        bloom_level = data.get('bloom_level', 'application')
        search_limit = data.get('search_limit', 10)
        score_threshold = data.get('score_threshold', 0.3)

        # Optional filters
        pdf_filename = data.get('pdf_filename')
        page_range = None
        if data.get('min_page') and data.get('max_page'):
            page_range = (data.get('min_page'), data.get('max_page'))

        # Validate parameters
        if num_questions < 1 or num_questions > 20:
            return {'error': 'num_questions must be between 1 and 20'}, 400

        if difficulty not in ['easy', 'medium', 'hard']:
            return {'error': 'difficulty must be easy, medium, or hard'}, 400

        if bloom_level not in ['knowledge', 'comprehension', 'application', 'analysis', 'synthesis', 'evaluation']:
            return {'error': 'invalid bloom_level'}, 400

        # Get MCQ generator
        generator = get_mcq_generator()

        # Run async MCQ generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                generator.generate_mcqs_from_search(
                    query=query,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    bloom_level=bloom_level,
                    search_limit=search_limit,
                    score_threshold=score_threshold,
                    pdf_filename=pdf_filename,
                    page_range=page_range
                )
            )
        finally:
            loop.close()

        # Return result
        if 'error' in result:
            return result, 400
        else:
            return {
                'status': 'success',
                'mcqs': result['mcqs'],
                'metadata': result['metadata'],
                'context_preview': result.get('context_preview'),
                'chunks_used': result.get('chunks_used', [])
            }, 200

    except Exception as e:
        app.logger.error(f"Error generating MCQs: {e}")
        return {'error': str(e)}, 500

@app.route('/generate-mcqs-simple', methods=['POST'])
def generate_mcqs_simple():
    """Simple MCQ generation endpoint with minimal parameters"""
    try:
        data = request.get_json()

        query = data.get('query')
        num_questions = data.get('num_questions', 3)

        if not query:
            return {'error': 'query is required'}, 400

        # Get MCQ generator
        generator = get_mcq_generator()

        # Run with default parameters
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                generator.generate_mcqs_from_search(
                    query=query,
                    num_questions=num_questions,
                    difficulty='medium',
                    bloom_level='application'
                )
            )
        finally:
            loop.close()

        if 'error' in result:
            return {'error': result['error']}, 400

        return {
            'mcqs': result['mcqs'],
            'total_questions': len(result['mcqs']),
            'query': query
        }, 200

    except Exception as e:
        app.logger.error(f"Error in simple MCQ generation: {e}")
        return {'error': str(e)}, 500

@app.route('/mcq-status', methods=['GET'])
def mcq_status():
    """Get MCQ generation system status"""
    try:
        # Check if Gemini API key is configured
        gemini_configured = bool(os.getenv('GEMINI_API_KEY'))

        # Check Qdrant connection
        try:
            generator = get_embedding_generator()
            pdfs = generator.get_pdfs_summary()
            qdrant_status = 'connected'
            total_chunks = sum(pdf.get('chunk_count', 0) for pdf in pdfs)
        except Exception as e:
            qdrant_status = f'error: {str(e)}'
            total_chunks = 0

        return {
            'gemini_configured': gemini_configured,
            'qdrant_status': qdrant_status,
            'total_chunks_available': total_chunks,
            'supported_difficulties': ['easy', 'medium', 'hard'],
            'supported_bloom_levels': ['knowledge', 'comprehension', 'application', 'analysis', 'synthesis', 'evaluation'],
            'max_questions_per_request': 20,
            'token_limit': 3000
        }, 200

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    # Disable debug mode to prevent TensorFlow-related restarts during chunking
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
