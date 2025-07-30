const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const redis = require("redis");
const winston = require("winston");
require("winston-daily-rotate-file");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, "logs");
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Configure Winston logger
const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: "document-qa-server" },
  transports: [
    // Error logs
    new winston.transports.DailyRotateFile({
      filename: path.join(logsDir, "error-%DATE%.log"),
      datePattern: "YYYY-MM-DD",
      level: "error",
      maxSize: "20m",
      maxFiles: "14d",
    }),
    // Combined logs
    new winston.transports.DailyRotateFile({
      filename: path.join(logsDir, "combined-%DATE%.log"),
      datePattern: "YYYY-MM-DD",
      maxSize: "20m",
      maxFiles: "14d",
    }),
    // HackRX specific logs
    new winston.transports.DailyRotateFile({
      filename: path.join(logsDir, "hackrx-%DATE%.log"),
      datePattern: "YYYY-MM-DD",
      maxSize: "20m",
      maxFiles: "30d",
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          return `${timestamp} [${level.toUpperCase()}]: ${message} ${
            Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ""
          }`;
        })
      ),
    }),
  ],
});

// Add console logging in development
if (process.env.NODE_ENV !== "production") {
  logger.add(
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      ),
    })
  );
}

// Initialize Redis client
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
  retry_strategy: (options) => {
    if (options.error && options.error.code === "ECONNREFUSED") {
      logger.error("Redis server connection refused");
      return new Error("Redis server connection refused");
    }
    if (options.total_retry_time > 1000 * 60 * 60) {
      return new Error("Redis retry time exhausted");
    }
    if (options.attempt > 10) {
      return undefined;
    }
    return Math.min(options.attempt * 100, 3000);
  },
});

redisClient.on("error", (err) => {
  logger.error("Redis Client Error", { error: err.message });
});

redisClient.on("connect", () => {
  logger.info("Redis Client Connected");
});

// Connect to Redis
(async () => {
  try {
    await redisClient.connect();
    logger.info("Redis connection established");
  } catch (error) {
    logger.error("Failed to connect to Redis", { error: error.message });
  }
})();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  req.requestId = requestId;
  req.startTime = startTime;

  logger.info("Incoming request", {
    requestId,
    method: req.method,
    url: req.url,
    userAgent: req.get("User-Agent"),
    ip: req.ip,
  });

  res.on("finish", () => {
    const duration = Date.now() - startTime;
    logger.info("Request completed", {
      requestId,
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
    });
  });

  next();
});

// Configure multer for file uploads
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

// Initialize Gemini with latest 2.5 Flash model
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// Initialize embeddings with latest model
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004", // Latest embedding model
});

// Store for FAISS instances (in production, use Redis or database)
const vectorStores = new Map();

// Cache configuration
const CACHE_TTL = 3600; // 1 hour in seconds
const EMBEDDING_CACHE_TTL = 86400; // 24 hours for embeddings

class DocumentProcessor {
  constructor() {
    // Optimized for speed - smaller chunks, less overlap
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600, // Smaller chunks for faster processing
      chunkOverlap: 50, // Minimal overlap for speed
    });
  }

  async extractTextFromFile(filePath, fileType) {
    try {
      logger.info("Extracting text from file", { filePath, fileType });
      
      if (fileType === "application/pdf") {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdfParse(dataBuffer);
        return data.text;
      } else if (
        fileType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        const result = await mammoth.extractRawText({ path: filePath });
        return result.value;
      } else if (fileType === "text/plain") {
        return fs.readFileSync(filePath, "utf8");
      } else {
        throw new Error(`Unsupported file type: ${fileType}`);
      }
    } catch (error) {
      logger.error("Error extracting text from file", { 
        filePath, 
        fileType, 
        error: error.message 
      });
      throw new Error(`Error extracting text: ${error.message}`);
    }
  }

  async downloadAndExtractFromUrl(url) {
    try {
      logger.info("Downloading document from URL", { url });
      
      // Check Redis cache first
      const cacheKey = `url_content:${Buffer.from(url).toString("base64")}`;
      try {
        const cachedContent = await redisClient.get(cacheKey);
        if (cachedContent) {
          logger.info("Found cached content for URL", { url });
          return cachedContent;
        }
      } catch (redisError) {
        logger.warn("Redis cache check failed", { error: redisError.message });
      }

      const response = await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 15000, // 15 second timeout
        maxContentLength: 10 * 1024 * 1024, // 10MB limit for speed
      });
      const contentType = response.headers["content-type"];

      // Create temporary file
      const tempDir = path.join(__dirname, "temp");
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir);
      }

      const tempFilePath = path.join(tempDir, `temp_${Date.now()}.pdf`);
      fs.writeFileSync(tempFilePath, response.data);

      logger.info("Extracting text from downloaded document");
      const text = await this.extractTextFromFile(
        tempFilePath,
        "application/pdf"
      );

      // Clean up temp file immediately
      fs.unlinkSync(tempFilePath);

      // Cache the extracted text
      try {
        await redisClient.setEx(cacheKey, CACHE_TTL, text);
        logger.info("Cached extracted text", { url, textLength: text.length });
      } catch (redisError) {
        logger.warn("Failed to cache extracted text", { error: redisError.message });
      }

      return text;
    } catch (error) {
      logger.error("Error downloading and extracting from URL", { 
        url, 
        error: error.message 
      });
      throw new Error(
        `Error downloading and extracting from URL: ${error.message}`
      );
    }
  }

  async processAndStoreDocument(documentSource, storeId, requestId) {
    try {
      let text;

      logger.info("Processing document", { documentSource: documentSource.substring(0, 100), storeId, requestId });
      
      // Check if documentSource is URL or file path
      if (documentSource.startsWith("http")) {
        text = await this.downloadAndExtractFromUrl(documentSource);
      } else {
        // Assume it's a file path
        text = await this.extractTextFromFile(
          documentSource,
          "application/pdf"
        );
      }

      logger.info("Splitting text into chunks", { 
        textLength: text.length, 
        storeId, 
        requestId 
      });

      // Split text into chunks
      const docs = await this.textSplitter.createDocuments([text]);

      // Save chunks to file for debugging
      const chunks = docs
        .map((doc, index) => `--- Chunk ${index} ---\n${doc.pageContent}\n\n`)
        .join("");

      const chunksFilePath = path.join(logsDir, `chunks_${requestId}.txt`);
      fs.writeFileSync(chunksFilePath, chunks, "utf8");
      logger.info("Chunks written to file", { 
        chunksFilePath, 
        chunksCount: docs.length,
        requestId 
      });

      // Check if embeddings are cached for similar content
      const contentHash = require("crypto")
        .createHash("md5")
        .update(text)
        .digest("hex");
      const embeddingCacheKey = `embeddings:${contentHash}`;

      let vectorStore;
      try {
        const cachedEmbeddings = await redisClient.get(embeddingCacheKey);
        if (cachedEmbeddings) {
          logger.info("Found cached embeddings", { contentHash, requestId });
          // Note: In a real implementation, you'd need to serialize/deserialize FAISS store
          // For now, we'll create new embeddings but this shows the caching pattern
        }
      } catch (redisError) {
        logger.warn("Embedding cache check failed", { error: redisError.message });
      }

      logger.info("Creating embeddings", { 
        chunksCount: docs.length, 
        storeId, 
        requestId 
      });

      // Create FAISS vector store with parallel processing
      vectorStore = await FaissStore.fromDocuments(docs, embeddings);

      // Store the vector store
      vectorStores.set(storeId, vectorStore);

      // Cache embedding metadata
      try {
        await redisClient.setEx(
          embeddingCacheKey,
          EMBEDDING_CACHE_TTL,
          JSON.stringify({ 
            storeId, 
            chunksCount: docs.length, 
            createdAt: new Date().toISOString() 
          })
        );
      } catch (redisError) {
        logger.warn("Failed to cache embedding metadata", { error: redisError.message });
      }

      logger.info("Document processed successfully", { 
        storeId, 
        chunksCount: docs.length, 
        requestId 
      });

      return {
        success: true,
        chunksCount: docs.length,
        storeId: storeId,
      };
    } catch (error) {
      logger.error("Error processing document", { 
        storeId, 
        requestId, 
        error: error.message,
        stack: error.stack 
      });
      throw new Error(`Error processing document: ${error.message}`);
    }
  }

  async answerQuestions(questions, storeId, requestId) {
    try {
      const vectorStore = vectorStores.get(storeId);
      if (!vectorStore) {
        throw new Error(
          "Document not found. Please upload the document first."
        );
      }

      logger.info("Processing questions", { 
        questionsCount: questions.length, 
        storeId, 
        requestId 
      });

      const answers = [];

      // Process questions in parallel for speed
      const answerPromises = questions.map(async (question, index) => {
        const questionId = `q${index + 1}`;
        logger.info("Processing question", { 
          questionId, 
          question: question.substring(0, 100), 
          requestId 
        });

        // Check cache for this specific question + document combination
        const questionCacheKey = `answer:${storeId}:${Buffer.from(question).toString("base64")}`;
        try {
          const cachedAnswer = await redisClient.get(questionCacheKey);
          if (cachedAnswer) {
            logger.info("Found cached answer", { questionId, requestId });
            return { index, answer: JSON.parse(cachedAnswer) };
          }
        } catch (redisError) {
          logger.warn("Answer cache check failed", { error: redisError.message });
        }

        // Retrieve only top 10 most relevant docs
        const relevantDocs = await vectorStore.similaritySearch(question, 10);

        // Prepare context from relevant documents
        const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

        // Optimized prompt for faster response
        const prompt = `
You are an expert assistant. Use ONLY the following context to answer the question.

Context:
${context}

Question:
${question}

Instructions:
- Give a one-sentence answer that includes all critical details from the context.
- Limit to 35 words max.
- Do NOT add any explanation or repeat the question.
- If the answer is not in the context, try harder and infer from the document. You cannot say document not found unless you are 200% sure its not there and you cant infer from rest of document 
- Respond only with a JSON array: ["answer"]

Answer:
`;

        const result = await model.generateContent(prompt);
        let rawText = result.response.text().trim();

        // Clean up code block formatting (remove ```json and ```)
        rawText = rawText.replace(/```json|```/g, "").trim();

        let parsedAnswer;
        try {
          parsedAnswer = JSON.parse(rawText);
        } catch (e) {
          parsedAnswer = [rawText];
        }

        // Cache the answer
        try {
          await redisClient.setEx(
            questionCacheKey,
            CACHE_TTL,
            JSON.stringify(parsedAnswer)
          );
          logger.info("Cached answer", { questionId, requestId });
        } catch (redisError) {
          logger.warn("Failed to cache answer", { error: redisError.message });
        }

        logger.info("Question processed", { 
          questionId, 
          answerLength: parsedAnswer[0]?.length || 0, 
          requestId 
        });

        return { index, answer: parsedAnswer };
      });

      // Wait for all answers to complete
      const results = await Promise.all(answerPromises);

      // Sort by original order
      results.sort((a, b) => a.index - b.index);

      logger.info("All questions processed", { 
        questionsCount: questions.length, 
        storeId, 
        requestId 
      });

      return results.flatMap((r) => r.answer);
    } catch (error) {
      logger.error("Error answering questions", { 
        storeId, 
        requestId, 
        error: error.message,
        stack: error.stack 
      });
      throw new Error(`Error answering questions: ${error.message}`);
    }
  }
}

const documentProcessor = new DocumentProcessor();

// Routes

// Health check
app.get("/health", async (req, res) => {
  const health = {
    status: "OK",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    redis: "disconnected",
  };

  try {
    await redisClient.ping();
    health.redis = "connected";
  } catch (error) {
    health.redis = "error";
  }

  res.json(health);
});

// Upload and process document (file upload)
app.post("/upload", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const storeId = `doc_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    logger.info("File upload started", { 
      filename: req.file.originalname,
      size: req.file.size,
      storeId,
      requestId: req.requestId 
    });

    const result = await documentProcessor.processAndStoreDocument(
      req.file.path,
      storeId,
      req.requestId
    );

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    logger.info("File upload completed", { 
      storeId, 
      chunksCount: result.chunksCount,
      requestId: req.requestId 
    });

    res.json({
      message: "Document processed successfully",
      storeId: storeId,
      chunksCount: result.chunksCount,
    });
  } catch (error) {
    logger.error("Upload error", { 
      error: error.message, 
      requestId: req.requestId,
      stack: error.stack 
    });
    res.status(500).json({ error: error.message });
  }
});

// Main endpoint matching the sample request format
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();
  const requestId = req.requestId;

  try {
    // Validate authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      logger.warn("Invalid authorization header", { requestId });
      return res
        .status(401)
        .json({ error: "Missing or invalid authorization header" });
    }

    const { documents, questions } = req.body;

    if (!documents || !questions || !Array.isArray(questions)) {
      logger.warn("Invalid request format", { 
        hasDocuments: !!documents,
        hasQuestions: !!questions,
        questionsIsArray: Array.isArray(questions),
        requestId 
      });
      return res.status(400).json({
        error:
          "Invalid request format. Expected documents (string) and questions (array)",
      });
    }

    // Generate a unique store ID for this request
    const storeId = `hackrx_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    logger.info("HackRX Request started", { 
      questionsCount: questions.length,
      documentLength: documents.length,
      storeId,
      requestId,
      authHeader: authHeader.substring(0, 20) + "..."
    });

    // Log the full request details to HackRX log
    const hackrxLogger = winston.createLogger({
      transports: [
        new winston.transports.File({
          filename: path.join(logsDir, `hackrx-${requestId}.log`),
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.printf(({ timestamp, level, message, ...meta }) => {
              return `${timestamp} [${level.toUpperCase()}]: ${message}\n${
                Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ""
              }`;
            })
          ),
        }),
      ],
    });

    hackrxLogger.info("HackRX Request Details", {
      requestId,
      storeId,
      timestamp: new Date().toISOString(),
      headers: req.headers,
      body: {
        documents: documents.substring(0, 500) + "...",
        questions: questions,
      },
    });

    // Process the document
    logger.info("Processing document", { 
      documentPreview: documents.substring(0, 100) + "...",
      storeId,
      requestId 
    });
    
    await documentProcessor.processAndStoreDocument(documents, storeId, requestId);

    // Answer the questions
    logger.info("Answering questions", { 
      questionsCount: questions.length,
      storeId,
      requestId 
    });
    
    const answers = await documentProcessor.answerQuestions(questions, storeId, requestId);

    // Clean up the vector store immediately for memory efficiency
    vectorStores.delete(storeId);

    const processingTime = Date.now() - startTime;
    
    logger.info("HackRX Request completed", { 
      processingTime: `${processingTime}ms`,
      answersCount: answers.length,
      storeId,
      requestId 
    });

    // Log the response to HackRX log
    hackrxLogger.info("HackRX Response", {
      requestId,
      storeId,
      processingTime,
      answers,
      timestamp: new Date().toISOString(),
    });

    // Return structured response matching the sample format
    res.json({
      answers: answers,
    });
  } catch (error) {
    const processingTime = Date.now() - startTime;
    
    logger.error("HackRX error", { 
      processingTime: `${processingTime}ms`,
      error: error.message,
      stack: error.stack,
      requestId 
    });
    
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

// Answer questions for existing document
app.post("/answer/:storeId", async (req, res) => {
  try {
    const { storeId } = req.params;
    const { questions } = req.body;

    if (!questions || !Array.isArray(questions)) {
      return res
        .status(400)
        .json({ error: "Questions must be provided as an array" });
    }

    logger.info("Answering questions for existing document", { 
      storeId, 
      questionsCount: questions.length,
      requestId: req.requestId 
    });

    const answers = await documentProcessor.answerQuestions(
      questions, 
      storeId, 
      req.requestId
    );

    res.json({
      storeId: storeId,
      answers: answers,
    });
  } catch (error) {
    logger.error("Answer error", { 
      storeId: req.params.storeId,
      error: error.message,
      requestId: req.requestId 
    });
    res.status(500).json({ error: error.message });
  }
});

// Get available document stores
app.get("/stores", (req, res) => {
  const stores = Array.from(vectorStores.keys());
  logger.info("Document stores requested", { 
    storesCount: stores.length,
    requestId: req.requestId 
  });
  res.json({ stores: stores, count: stores.length });
});

// Delete a document store
app.delete("/stores/:storeId", (req, res) => {
  const { storeId } = req.params;

  if (vectorStores.has(storeId)) {
    vectorStores.delete(storeId);
    logger.info("Document store deleted", { 
      storeId,
      requestId: req.requestId 
    });
    res.json({ message: "Document store deleted successfully" });
  } else {
    logger.warn("Document store not found for deletion", { 
      storeId,
      requestId: req.requestId 
    });
    res.status(404).json({ error: "Document store not found" });
  }
});

// Clear cache endpoint
app.post("/cache/clear", async (req, res) => {
  try {
    await redisClient.flushAll();
    logger.info("Cache cleared", { requestId: req.requestId });
    res.json({ message: "Cache cleared successfully" });
  } catch (error) {
    logger.error("Error clearing cache", { 
      error: error.message,
      requestId: req.requestId 
    });
    res.status(500).json({ error: "Failed to clear cache" });
  }
});

// Get cache stats
app.get("/cache/stats", async (req, res) => {
  try {
    const info = await redisClient.info();
    const keyCount = await redisClient.dbSize();
    
    res.json({
      connected: true,
      keyCount,
      info: info.split('\r\n').reduce((acc, line) => {
        const [key, value] = line.split(':');
        if (key && value) acc[key] = value;
        return acc;
      }, {}),
    });
  } catch (error) {
    logger.error("Error getting cache stats", { 
      error: error.message,
      requestId: req.requestId 
    });
    res.status(500).json({ 
      connected: false, 
      error: error.message 
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error("Unhandled error", { 
    error: error.message,
    stack: error.stack,
    requestId: req.requestId 
  });
  res.status(500).json({
    error: "Internal server error",
    message: error.message,
  });
});

// 404 handler
app.use((req, res) => {
  logger.warn("Endpoint not found", { 
    method: req.method,
    url: req.url,
    requestId: req.requestId 
  });
  res.status(404).json({ error: "Endpoint not found" });
});

// Start server
app.listen(port, () => {
  logger.info("Server started", { port });
  console.log(`Document Q&A Server running on port ${port}`);
  console.log("Available endpoints:");
  console.log("  POST /hackrx/run - Main endpoint (matches sample format)");
  console.log("  POST /upload - Upload document file");
  console.log("  POST /answer/:storeId - Answer questions for existing document");
  console.log("  GET /stores - List document stores");
  console.log("  DELETE /stores/:storeId - Delete document store");
  console.log("  POST /cache/clear - Clear Redis cache");
  console.log("  GET /cache/stats - Get cache statistics");
  console.log("  GET /health - Health check");
  console.log(`Logs are being saved to: ${logsDir}`);
});

// Graceful shutdown
process.on("SIGINT", async () => {
  logger.info("Shutting down gracefully...");
  console.log("\nShutting down gracefully...");
  
  try {
    await redisClient.quit();
    logger.info("Redis connection closed");
  } catch (error) {
    logger.error("Error closing Redis connection", { error: error.message });
  }
  
  process.exit(0);
});

process.on("uncaughtException", (error) => {
  logger.error("Uncaught Exception", { error: error.message, stack: error.stack });
  process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection", { reason, promise });
});

module.exports = app;