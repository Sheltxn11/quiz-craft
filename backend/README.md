# Quiz Craft Backend - Phase 3: Text Processing Implementation

This backend implements Phases 2 & 3 of the Quiz Craft application, featuring text extraction from PDFs using Mistral's Document OCR API and intelligent text processing and structuring.

## Quick Start

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Start the server
py server.py

# 3. Test OCR functionality
py test_ocr.py
```

The server will be available at `http://localhost:5000`

## Features

### Phase 2: OCR Processing
- **PDF Upload**: Upload PDFs to Azure Blob Storage
- **OCR Processing**: Extract text from PDFs using Mistral's OCR API
- **Blob Storage Integration**: Secure retrieval from Azure Blob Storage with authentication
- **Base64 Encoding**: Converts PDFs to base64 for Mistral API compatibility

### Phase 3: Text Processing & Structuring
- **Page Range Filtering**: Process only specified page ranges (from/to parameters)
- **Text Cleaning**: Remove headers, footers, and OCR artifacts
- **Content Structuring**: Parse text into meaningful segments (titles, headers, paragraphs)
- **Metadata Extraction**: Track page numbers, section titles, and document structure
- **Intelligent Parsing**: Detect titles, headers, and content hierarchy using heuristics

### General Features
- **Job Tracking**: Track processing status with unique job IDs
- **Error Handling**: Comprehensive error handling for API failures
- **Rate Limiting**: Built-in handling for API rate limits
- **Temporary Storage**: In-memory storage for processed text (1-hour expiry)
- **Multiple Output Formats**: Raw text, structured data, and processed text

## Setup

### 1. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the backend directory:

```bash
cp .env.example .env
```

Edit `.env` and add your Mistral API key:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Get Mistral API Key

1. Visit [Mistral AI Console](https://console.mistral.ai/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## API Endpoints

### 1. Upload PDF and Start OCR Processing

**POST** `/generate-mcq`

Upload a PDF file and automatically start OCR processing.

**Request:**
- Form data with PDF file
- Optional: `from`, `to`, `num_questions` parameters

**Response:**
```json
{
  "status": "received",
  "job_id": "uuid-string",
  "blob_url": "https://..."
}
```

### 2. Check OCR Status

**GET** `/ocr-status/<job_id>`

Check the processing status of an OCR job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing|completed|failed",
  "timestamp": 1234567890,
  "text_length": 1500  // Only for completed jobs
}
```

### 3. Get Extracted Text

**GET** `/extracted-text/<job_id>`

Retrieve the extracted text from a completed OCR job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "text": "Extracted text content...",
  "timestamp": 1234567890
}
```

### 4. Get Structured Text

**GET** `/structured-text/<job_id>`

Retrieve the structured text data with metadata for a completed job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "structured_data": {
    "segments": [...],
    "metadata": {...}
  },
  "timestamp": 1234567890
}
```

### 5. Direct OCR + Text Processing

**POST** `/process-ocr`

Process OCR and text structuring directly with a blob URL (synchronous).

**Request:**
```json
{
  "blob_url": "https://...",
  "page_from": 1,
  "page_to": 5
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "result": {
    "segments": [...],
    "metadata": {...},
    "processed_text": "Clean structured text..."
  },
  "status": "completed"
}
```

## Structured Data Format

The text processing engine returns structured data with the following format:

### Segment Structure
```json
{
  "content": "Text content of the segment",
  "page_number": 1,
  "type": "title|header|paragraph|list",
  "section_title": "Parent section title",
  "metadata": {}
}
```

### Metadata Structure
```json
{
  "total_pages": 5,
  "total_segments": 42,
  "segment_types": {
    "title": 8,
    "header": 15,
    "paragraph": 18,
    "list": 1
  },
  "pages_processed": [1, 2, 3],
  "sections": [
    {"title": "Introduction", "page": 1},
    {"title": "Methodology", "page": 2}
  ],
  "word_count": 1250,
  "character_count": 7500
}
```

## Usage Examples

### Using curl

```bash
# Check OCR status
curl http://localhost:5000/ocr-status/your-job-id

# Get extracted text
curl http://localhost:5000/extracted-text/your-job-id

# Direct OCR processing
curl -X POST http://localhost:5000/process-ocr \
  -H "Content-Type: application/json" \
  -d '{"blob_url": "https://your-pdf-url.com/file.pdf"}'
```

### Using Python

```python
import requests

# Check status
response = requests.get('http://localhost:5000/ocr-status/your-job-id')
status = response.json()

# Get extracted text
response = requests.get('http://localhost:5000/extracted-text/your-job-id')
text_data = response.json()
```

## Testing

Run the test scripts to verify OCR functionality:

```bash
# Test with external PDF
py test_ocr.py

# Test with Azure Blob Storage
py test_blob_ocr.py

# Test Phase 3 text processing
py test_text_processing.py
```

This will test:
- OCR processing with a sample PDF
- Azure Blob Storage integration
- Text cleaning and structuring
- Page range filtering
- Status endpoint functionality
- Error handling

## Error Handling

The implementation includes comprehensive error handling for:

- **Missing API Key**: Returns appropriate error messages
- **Network Failures**: Handles timeouts and connection errors
- **API Rate Limits**: Implements retry logic (can be enhanced)
- **Invalid PDFs**: Handles malformed or unsupported files
- **Job Expiry**: Automatically cleans up old jobs after 1 hour
- **Blob Access Issues**: Falls back to direct HTTP requests if Azure auth fails

## Troubleshooting

### Common Issues

**1. "File could not be fetched from url" Error**
- **Fixed**: The system now downloads PDFs from Azure Blob Storage and converts them to base64
- **Solution**: PDFs are retrieved using authenticated Azure client and sent as base64 to Mistral

**2. "Mistral API key not configured" Error**
- **Cause**: Missing or invalid MISTRAL_API_KEY in .env file
- **Solution**: Ensure your .env file contains: `MISTRAL_API_KEY=your_actual_key`

**3. "Failed to retrieve PDF from blob storage" Error**
- **Cause**: Azure connection string or blob access issues
- **Solution**: Check Azure connection string and blob permissions

**4. OCR Processing Takes Too Long**
- **Cause**: Large PDF files or network latency
- **Solution**: Increase timeout values in requests (currently 30s for download, 120s for OCR)

## Production Considerations

For production deployment, consider:

1. **Database Storage**: Replace in-memory storage with a proper database
2. **Task Queue**: Use Celery or similar for background OCR processing
3. **Caching**: Implement Redis for better performance
4. **Rate Limiting**: Add proper rate limiting middleware
5. **Monitoring**: Add logging and monitoring for OCR jobs
6. **Security**: Implement authentication and authorization

## Limitations

- **File Size**: Mistral OCR supports files up to 50 MB
- **Page Limit**: Maximum 1,000 pages per document
- **Memory Storage**: Current implementation uses in-memory storage
- **Synchronous Processing**: OCR processing blocks the request thread

## Next Steps

Phase 3 will implement:
- Question generation from extracted text
- Integration with AI models for MCQ creation
- Enhanced error handling and retry mechanisms
