# Document Q&A System with Multi-Format Support

A high-performance document question-answering system that supports multiple file formats including PDFs, Word documents, Excel spreadsheets, and images with OCR processing.

## 🚀 Features

### Core Capabilities
- **Multi-Format Support**: PDF, Word (.doc/.docx), Excel (.xls/.xlsx), Images (JPEG, PNG, GIF, BMP, TIFF)
- **OCR Processing**: Automatic text extraction from images using Tesseract.js
- **High-Speed Processing**: Optimized for fast response times with intelligent caching
- **AI-Powered Answers**: Uses Google Gemini 2.5 Flash with Groq fallback for reliability
- **Parallel Processing**: Handles multiple questions simultaneously
- **Smart Caching**: Caches processed files and OCR results for faster subsequent requests

### Performance Optimizations
- **Intelligent Caching**: File content, OCR results, and vector stores are cached
- **Parallel Processing**: Multiple questions processed simultaneously
- **Image Preprocessing**: Sharp library optimizes images for better OCR accuracy
- **Connection Pooling**: Optimized HTTP/HTTPS agents for faster downloads
- **Memory Management**: Automatic cleanup of temporary files

## 📋 Supported File Types

| Format | Extensions | Processing Method |
|--------|------------|-------------------|
| PDF | `.pdf` | Direct text extraction |
| Word | `.doc`, `.docx` | Mammoth library |
| Excel | `.xls`, `.xlsx` | XLSX library with multi-sheet support |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff` | OCR with Tesseract.js |

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables in `.env`:
   ```env
   PORT=3000
   GOOGLE_API_KEY=your_gemini_api_key
   GOOGLE_EMBEDDING_KEY=your_google_embedding_key
   GROQ_API_KEY=your_groq_api_key
   REDIS_URL=redis://localhost:6379
   ```

4. Start the server:
   ```bash
   npm start
   ```

## 📡 API Endpoints

### 1. File Upload and Processing
**POST** `/hackrx/upload`

Upload and process files with automatic format detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The document file to process
  - `questions`: JSON array of questions

**Example:**
```bash
curl -X POST http://localhost:3000/hackrx/upload \
  -F "file=@document.pdf" \
  -F "questions=[\"What is the main topic?\", \"What are the key points?\"]"
```

**Response:**
```json
{
  "answers": ["Answer 1", "Answer 2"],
  "metadata": {
    "file_name": "document.pdf",
    "file_type": "pdf",
    "file_size_mb": "2.5",
    "text_length": 15000,
    "processing_time_ms": 3500,
    "extraction_time_ms": 800,
    "vector_time_ms": 1200,
    "questions_processed": 2,
    "avg_time_per_question_ms": "1750.0"
  }
}
```

### 2. PDF from URL (Existing)
**POST** `/hackrx/run`

Process PDFs from URLs (existing functionality).

**Request:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is the main topic?", "What are the key points?"]
}
```

### 3. Health Check
**GET** `/health`

Check system status and capabilities.

**Response:**
```json
{
  "status": "OK",
  "file_processing": {
    "supported_formats": [
      "PDF files",
      "Word documents (.doc/.docx)",
      "Excel spreadsheets (.xls/.xlsx)",
      "Images with OCR (JPEG, PNG, GIF, BMP, TIFF)"
    ],
    "ocr_engine": "Tesseract.js with image preprocessing",
    "max_file_size": "50MB",
    "processing_cache": "Enabled for all file types"
  }
}
```

### 4. Cache Management
**POST** `/cache/clear`

Clear all caches (file processing, OCR, vector stores, etc.).

## 🔧 Configuration

### File Upload Limits
- **Maximum file size**: 50MB
- **Supported formats**: PDF, Word, Excel, Images
- **OCR languages**: English (configurable)

### Performance Settings
- **Chunk size**: 1200 characters
- **Chunk overlap**: 300 characters
- **Vector search results**: 8 documents
- **Cache TTL**: 1 hour for vector stores

## 🧪 Testing

Use the provided test script to verify functionality:

```bash
node test-file-upload.js
```

The test script will check for sample files:
- `test.pdf` - PDF processing
- `test.docx` - Word document processing
- `test.xlsx` - Excel spreadsheet processing
- `test-image.jpg` - Image OCR processing

## 📊 Performance Metrics

### Typical Processing Times
- **PDF files**: 1-3 seconds
- **Word documents**: 1-2 seconds
- **Excel files**: 2-4 seconds
- **Images with OCR**: 5-15 seconds (depending on image complexity)

### Caching Benefits
- **First request**: Full processing time
- **Subsequent requests**: 80-90% faster due to caching
- **OCR results**: Cached permanently for identical images

## 🔍 OCR Processing

### Image Preprocessing
- **Resize**: Maximum 2000x2000 pixels
- **Sharpen**: Enhanced edge detection
- **Normalize**: Improved contrast
- **Format**: Converted to PNG for optimal OCR

### OCR Features
- **Language**: English (default)
- **Progress tracking**: Real-time processing updates
- **Error handling**: Graceful fallback for failed OCR
- **Result caching**: Permanent storage of OCR results

## 🚨 Error Handling

The system includes comprehensive error handling:

- **File type validation**: Automatic format detection
- **Size limits**: 50MB maximum file size
- **Processing errors**: Graceful fallback and cleanup
- **OCR failures**: Detailed error messages
- **Memory management**: Automatic temporary file cleanup

## 🔄 Caching Strategy

### Multi-Level Caching
1. **File Content Cache**: Extracted text from documents
2. **OCR Cache**: Text extracted from images
3. **Vector Store Cache**: Embeddings and search indices
4. **Answer Cache**: Processed question-answer pairs
5. **Redis Cache**: Distributed caching for scalability

### Cache Invalidation
- **Automatic TTL**: 1 hour for vector stores
- **Manual clearing**: `/cache/clear` endpoint
- **Memory management**: Automatic cleanup of old entries

## 📈 Scalability

### Performance Optimizations
- **Parallel processing**: Multiple questions handled simultaneously
- **Connection pooling**: Optimized HTTP/HTTPS agents
- **Memory efficiency**: Streaming file processing
- **Cache sharing**: Redis for distributed deployments

### Resource Management
- **File cleanup**: Automatic removal of temporary files
- **Memory monitoring**: Built-in memory usage tracking
- **Error recovery**: Graceful handling of processing failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

ISC License

## 🆘 Support

For issues and questions:
1. Check the health endpoint: `GET /health`
2. Review error logs in the console
3. Clear caches if needed: `POST /cache/clear`
4. Ensure all environment variables are set correctly
