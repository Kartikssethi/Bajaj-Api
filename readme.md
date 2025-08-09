# Bajaj API - Document Processing Service

A powerful document processing and question-answering API built with Node.js, Express, and AI services.

## 🚀 Features

- **Multi-format Support**: PDF, Word, Excel, PowerPoint, CSV, Images, Text, Markdown, HTML, JSON, XML, and ZIP archives
- **AI-Powered Processing**: Uses Google Gemini and Groq for intelligent document analysis
- **File Upload & URL Processing**: Support for both file uploads and URL-based document processing
- **Caching System**: Intelligent caching for improved performance
- **Rate Limiting**: Built-in protection against abuse
- **Comprehensive Error Handling**: Detailed error messages and logging
- **Modular Architecture**: Clean, maintainable code structure

## 📋 Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm
- Environment variables configured

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Bajaj-Api
```

2. Install dependencies:
```bash
pnpm install
# or
npm install
```

3. Create a `.env` file with your configuration:
```env
PORT=3000
NODE_ENV=development
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

4. Start the server:
```bash
# Development mode
pnpm dev

# Production mode
pnpm start
```

## 🏗️ Project Structure

```
src/
├── config/          # Configuration files
├── middleware/      # Express middleware
├── models/          # Data models
├── routes/          # API route handlers
├── services/        # Business logic services
├── types/           # TypeScript type definitions
└── utils/           # Utility functions
```

## 📚 API Endpoints

### Main Processing Endpoint

#### `POST /api/hackrx/run`

Process documents and answer questions.

**Request Body (File Upload):**
```json
{
  "file": "<file>",
  "questions": ["What is the main topic?", "Who is the author?"]
}
```

**Request Body (URL Processing):**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is the main topic?", "Who is the author?"]
}
```

**Response:**
```json
{
  "answers": ["Answer 1", "Answer 2"]
}
```

### Health Check

#### `GET /health`
Basic health status.

#### `GET /health/detailed`
Detailed system information.

### Cache Management

#### `GET /api/cache/stats`
Get cache statistics.

#### `GET /api/cache/keys`
List all cache keys.

#### `GET /api/cache/:key`
Get specific cache entry.

#### `DELETE /api/cache/:key`
Delete specific cache entry.

#### `DELETE /api/cache/clear`
Clear all cache.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 3000 |
| `NODE_ENV` | Environment mode | development |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `ALLOWED_ORIGINS` | CORS allowed origins | * |

### File Upload Limits

- **Maximum file size**: 50MB
- **Supported formats**: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX, CSV, Images, Text, Markdown, HTML, JSON, XML, ZIP

## 🚀 Usage Examples

### Using cURL

**File Upload:**
```bash
curl -X POST http://localhost:3000/api/hackrx/run \
  -F "file=@document.pdf" \
  -F "questions=[\"What is this document about?\"]"
```

**URL Processing:**
```bash
curl -X POST http://localhost:3000/api/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

### Using JavaScript/Fetch

```javascript
// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('questions', JSON.stringify(['What is this about?']));

const response = await fetch('/api/hackrx/run', {
  method: 'POST',
  body: formData
});

// URL processing
const response = await fetch('/api/hackrx/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    documents: 'https://example.com/document.pdf',
    questions: ['What is this about?']
  })
});
```

## 🔒 Security Features

- **Helmet.js**: Security headers
- **Rate Limiting**: 100 requests per 15 minutes per IP
- **CORS Protection**: Configurable origin restrictions
- **Input Validation**: File type and size validation
- **Error Sanitization**: Safe error messages in production

## 📊 Performance

- **Caching**: Intelligent caching for processed documents
- **Compression**: Response compression for large payloads
- **Async Processing**: Non-blocking document processing
- **Memory Management**: Efficient file handling and cleanup

## 🐛 Troubleshooting

### Common Issues

1. **File Upload Fails**
   - Check file size (max 50MB)
   - Verify file format is supported
   - Ensure proper multipart/form-data encoding

2. **API Key Errors**
   - Verify environment variables are set
   - Check API key validity
   - Ensure proper permissions

3. **Processing Timeouts**
   - Large documents may take longer
   - Check network connectivity
   - Verify AI service availability

### Logs

Check console output for detailed error information and request tracking.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the ISC License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Built with ❤️ for the Bajaj Hackathon**
