# RAG Implementation Documentation

## Overview

The system has been enhanced with a comprehensive RAG (Retrieval-Augmented Generation) component that uses FAISS vector store for document embeddings, proper Redis caching, and supports all file types for ingestion.

## Key Features

### 🧠 RAG Service (`src/services/RAGService.js`)

- **FAISS Vector Store**: Uses `@langchain/community/vectorstores/faiss` for efficient similarity search
- **Google Embeddings**: Powered by `models/text-embedding-004` 
- **Text Chunking**: Recursive character text splitter with optimized chunk size (1000) and overlap (200)
- **Multi-layer Caching**: Memory, Redis, and disk persistence
- **Automatic Cleanup**: Old vector stores are cleaned up automatically

### 🔄 Enhanced LLMService (`src/services/LLMService.js`)

- **RAG Integration**: Seamlessly integrates with RAG service for context retrieval
- **Fallback Mode**: Gracefully falls back to direct text processing if RAG fails
- **Enhanced Context**: Uses similarity search to find most relevant document chunks
- **Smart Caching**: Caches both embeddings and vector stores for improved performance

### 📁 Document Processing (`src/routes/hackrx.js`)

- **Universal Support**: All existing file types (PDF, Word, Excel, PowerPoint, Images, etc.) now use RAG
- **Automatic Ingestion**: Documents are automatically processed and stored in vector database
- **Performance Logging**: Detailed timing information for RAG vs fallback modes

## Architecture

```
Document Upload/URL
        ↓
File Processing (existing)
        ↓
Text Extraction
        ↓
RAG Service
        ↓
┌─────────────────┐
│ Text Chunking   │
│ (1000 chars)    │
└─────────────────┘
        ↓
┌─────────────────┐
│ Embeddings      │
│ (Google API)    │
└─────────────────┘
        ↓
┌─────────────────┐
│ FAISS Vector    │
│ Store Creation  │
└─────────────────┘
        ↓
┌─────────────────┐
│ Caching Layer   │
│ Memory/Redis/   │
│ Disk            │
└─────────────────┘
        ↓
Question Processing
        ↓
┌─────────────────┐
│ Similarity      │
│ Search (k=5)    │
└─────────────────┘
        ↓
┌─────────────────┐
│ Context         │
│ Assembly        │
└─────────────────┘
        ↓
LLM Processing (Gemini/Groq)
        ↓
Answer Generation
```

## Storage Structure

### FAISS Stores
- **Location**: `./faiss-stores/`
- **Format**: `{documentHash}.faiss` and `{documentHash}.faiss.pkl`
- **Persistence**: Automatic save/load from disk

### Redis Cache Keys
- **Vector Stores**: `vectorstore:{documentHash}` (TTL: 2 hours)
- **Embeddings**: `embedding:{textHash}` (TTL: 2 hours)
- **Buffers**: `url_buffer:{urlHash}` (TTL: 4 hours)

### Memory Cache
- Vector stores, embeddings, text content, and answer caches

## API Endpoints

### Document Processing
```
POST /hackrx/run
```
- Enhanced with RAG processing
- Automatic vector store creation
- Intelligent context retrieval

### Cache Management
```
GET /cache/stats
```
- Includes RAG storage statistics
- Memory, Redis, and disk usage

```
DELETE /cache/clear
```
- Clears all caches including RAG

```
GET /cache/rag/stats
```
- RAG-specific storage statistics

```
DELETE /cache/rag/clear
```
- Clear only RAG caches and vector stores

```
DELETE /cache/rag/cleanup?maxAge=604800000
```
- Cleanup old FAISS stores (default: 7 days)

## Configuration

### Environment Variables
```env
GOOGLE_API_KEY=your_google_api_key    # Required for embeddings
REDIS_URL=redis://localhost:6379     # Optional, falls back to memory
```

### Tunable Parameters

#### RAGService
- `chunkSize`: 1000 characters
- `chunkOverlap`: 200 characters  
- `maxChunks`: 5 for context retrieval
- `cacheTimeout`: 2 hours for Redis

#### Text Splitter
- Separators: `["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]`

## Error Handling

### Graceful Degradation
1. **RAG Failure**: Falls back to direct text processing
2. **Embedding Service Unavailable**: Error with clear message
3. **Redis Unavailable**: Uses memory cache only
4. **FAISS Store Corruption**: Recreates vector store

### Logging
- Detailed request-level logging with unique request IDs
- Performance metrics for each processing stage
- Error context with specific failure reasons

## Performance Optimizations

### Caching Strategy
1. **Memory Cache**: Instant access to recent vector stores
2. **Redis Cache**: Shared cache across instances
3. **Disk Persistence**: Long-term storage for FAISS indexes

### Context Retrieval
- Optimized similarity search with k=5 chunks
- Intelligent context assembly
- Minimal token usage for LLM calls

### Chunking Strategy
- Optimized chunk size for semantic coherence
- Smart overlap to maintain context continuity
- Separator hierarchy for clean splits

## Monitoring

### Metrics Available
- Vector store cache hit/miss rates
- Embedding generation times
- FAISS search performance
- Storage usage (memory/disk/Redis)
- Processing mode (RAG vs fallback)

### Health Checks
- RAG service initialization status
- Embedding service availability
- Redis connectivity
- FAISS store integrity

## Migration Notes

### From Previous Implementation
- Seamless migration from direct text processing
- Existing caches are preserved
- Backward compatibility maintained
- No breaking changes to API

### Rollback Strategy
- Set `GOOGLE_API_KEY=""` to disable RAG
- System automatically falls back to direct processing
- No data loss or corruption

## File Type Support

All existing file types continue to work with RAG:

- **PDF**: Direct text extraction → RAG processing
- **Word** (.doc/.docx): Mammoth extraction → RAG processing  
- **Excel** (.xls/.xlsx): Sheet data → RAG processing
- **PowerPoint** (.ppt/.pptx): Slide content + OCR → RAG processing
- **Images**: OCR with Gemini/Groq → RAG processing
- **CSV**: Structured data → RAG processing
- **Text/Markdown/HTML/JSON/XML**: Direct content → RAG processing
- **ZIP**: Recursive extraction → RAG processing for each file

## Troubleshooting

### Common Issues

1. **"Embeddings service not available"**
   - Check `GOOGLE_API_KEY` environment variable
   - Verify Google API quota

2. **"Vector store creation failed"**
   - Check disk space in `faiss-stores/` directory
   - Verify write permissions
   - Check memory usage

3. **Slow performance**
   - Monitor Redis connection
   - Check FAISS store size
   - Review chunk count and size

4. **Memory usage**
   - Clear caches via `/cache/clear`
   - Cleanup old stores via `/cache/rag/cleanup`
   - Monitor vector store cache size

### Debug Commands

```bash
# Check RAG statistics
curl http://localhost:3000/cache/rag/stats

# Clear RAG caches
curl -X DELETE http://localhost:3000/cache/rag/clear

# Cleanup old stores
curl -X DELETE "http://localhost:3000/cache/rag/cleanup?maxAge=86400000"
```

## Future Enhancements

### Planned Features
- Multiple embedding models support
- Custom chunk size per document type
- Vector store compression
- Distributed FAISS with multiple replicas
- Advanced similarity search algorithms
- Document metadata enrichment

### Performance Improvements
- Async embedding generation
- Batch processing for multiple documents
- Streaming vector store updates
- Smart cache preloading
- Vector quantization for reduced memory usage
