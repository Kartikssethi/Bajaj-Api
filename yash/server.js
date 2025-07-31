const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { Groq } = require("groq-sdk");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const redis = require("redis");
const winston = require("winston");
require("winston-daily-rotate-file");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

// Create logs and faiss-stores directories if they don't exist
const logsDir = path.join(__dirname, "logs");
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

const faissStoresDir = path.join(__dirname, "faiss-stores");
if (!fs.existsSync(faissStoresDir)) {
  fs.mkdirSync(faissStoresDir, { recursive: true });
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
  const requestId = `req_${Date.now()}_${Math.random()
    .toString(36)
    .substr(2, 9)}`;

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

// Initialize Groq client
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

// Initialize embeddings with Google (keeping for embeddings as Groq doesn't provide embeddings)
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004", // Latest embedding model
});

// Store for FAISS instances (in-memory cache for currently active stores)
// This supplements the disk persistence
const vectorStores = new Map();

// Cache configuration
const CACHE_TTL = 3600; // 1 hour in seconds for answers and URL content

class DocumentProcessor {
  constructor() {
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
        error: error.message,
      });
      throw new Error(`Error extracting text: ${error.message}`);
    }
  }

  async downloadAndExtractFromUrl(url, requestId) {
    try {
      logger.info("Downloading document from URL", { url, requestId });

      // Check Redis cache for content first
      const cacheKey = `url_content:${Buffer.from(url).toString("base64")}`;
      let cachedContent;
      try {
        cachedContent = await redisClient.get(cacheKey);
        if (cachedContent) {
          logger.info("Found cached content for URL", { url, requestId });
          return cachedContent;
        }
      } catch (redisError) {
        logger.warn("Redis cache check for URL content failed", {
          error: redisError.message,
          requestId,
        });
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
      // Infer file extension from content-type header if possible
      let fileExtension = ".pdf"; // Default to PDF
      if (
        contentType.includes("application/msword") ||
        contentType.includes(
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
      ) {
        fileExtension = ".docx";
      } else if (contentType.includes("text/plain")) {
        fileExtension = ".txt";
      }
      const finalTempFilePath = path.join(
        tempDir,
        `temp_${Date.now()}${fileExtension}`
      );

      fs.writeFileSync(finalTempFilePath, response.data);

      logger.info("Extracting text from downloaded document", {
        requestId,
        finalTempFilePath,
        contentType,
      });
      const text = await this.extractTextFromFile(
        finalTempFilePath,
        contentType // Use actual content type for extraction
      );

      // Clean up temp file immediately
      fs.unlinkSync(finalTempFilePath);

      // Cache the extracted text
      try {
        await redisClient.setEx(cacheKey, CACHE_TTL, text);
        logger.info("Cached extracted text from URL", {
          url,
          textLength: text.length,
          requestId,
        });
      } catch (redisError) {
        logger.warn("Failed to cache extracted text from URL", {
          error: redisError.message,
          requestId,
        });
      }

      return text;
    } catch (error) {
      logger.error("Error downloading and extracting from URL", {
        url,
        error: error.message,
        requestId,
      });
      throw new Error(
        `Error downloading and extracting from URL: ${error.message}`
      );
    }
  }

  async processAndStoreDocument(documentSource, requestId, fileType = null) {
    try {
      let text;
      let actualFileType = fileType; // Use provided fileType for uploads

      logger.info("Processing document source", {
        documentSource: documentSource.substring(0, 100),
        requestId,
        initialFileTypeHint: fileType,
      });

      if (documentSource.startsWith("http")) {
        // If it's a URL, download and extract
        text = await this.downloadAndExtractFromUrl(documentSource, requestId);
      } else if (actualFileType) {
        // This branch is explicitly for file uploads
        text = await this.extractTextFromFile(documentSource, actualFileType);
      } else {
        // This branch is for raw text content
        text = documentSource;
      }

      // Ensure 'text' is actually a string before proceeding
      if (typeof text !== "string") {
        const errorMessage = `Invalid document content type. Expected string, got ${typeof text}. Value: ${text}`;
        logger.error(errorMessage, {
          documentSource,
          requestId,
          originalFileTypeHint: fileType,
        });
        throw new Error(errorMessage);
      }

      // Calculate content hash for persistent caching
      const contentHash = require("crypto")
        .createHash("md5")
        .update(text)
        .digest("hex");
      const faissStorePath = path.join(faissStoresDir, `${contentHash}.faiss`);

      // Check if FAISS store is already on disk
      if (fs.existsSync(faissStorePath)) {
        logger.info("Loading FAISS store from disk cache", {
          contentHash,
          faissStorePath,
          requestId,
        });
        const vectorStore = await FaissStore.load(faissStorePath, embeddings);
        // Add to in-memory map for quick access during current session
        vectorStores.set(contentHash, vectorStore);
        return {
          success: true,
          chunksCount: "loaded_from_cache", // Indicate it was loaded
          storeId: contentHash,
        };
      }

      logger.info("Splitting text into chunks", {
        textLength: text.length,
        requestId,
      });

      const docs = await this.textSplitter.createDocuments([text]);

      // Save chunks to file for debugging (optional, can be removed in production)
      const chunks = docs
        .map((doc, index) => `--- Chunk ${index} ---\n${doc.pageContent}\n\n`)
        .join("");
      const chunksFilePath = path.join(logsDir, `chunks_${requestId}.txt`);
      fs.writeFileSync(chunksFilePath, chunks, "utf8");
      logger.info("Chunks written to file", {
        chunksFilePath,
        chunksCount: docs.length,
        requestId,
      });

      logger.info("Creating embeddings and FAISS store", {
        chunksCount: docs.length,
        requestId,
      });

      const vectorStore = await FaissStore.fromDocuments(docs, embeddings);

      // Save the FAISS store to disk for persistence
      await vectorStore.save(faissStorePath);
      logger.info("FAISS store saved to disk", {
        faissStorePath,
        contentHash,
        requestId,
      });

      // Add to in-memory map for quick access during current session
      vectorStores.set(contentHash, vectorStore);

      logger.info("Document processed successfully", {
        storeId: contentHash,
        chunksCount: docs.length,
        requestId,
      });

      return {
        success: true,
        chunksCount: docs.length,
        storeId: contentHash,
      };
    } catch (error) {
      logger.error("Error processing document", {
        documentSource: documentSource.substring(0, 100),
        requestId,
        error: error.message,
        stack: error.stack,
      });
      throw new Error(`Error processing document: ${error.message}`);
    }
  }

  async answerQuestions(questions, storeId, requestId) {
    try {
      let vectorStore = vectorStores.get(storeId);
      if (!vectorStore) {
        // If the store is not in memory, try to load it from disk based on contentHash (storeId)
        const faissStorePath = path.join(faissStoresDir, `${storeId}.faiss`);
        if (fs.existsSync(faissStorePath)) {
          logger.info("Loading FAISS store for answering from disk", {
            storeId,
            faissStorePath,
            requestId,
          });
          vectorStore = await FaissStore.load(faissStorePath, embeddings);
          vectorStores.set(storeId, vectorStore); // Add to in-memory map
          logger.info("FAISS store loaded for answering", {
            storeId,
            requestId,
          });
        } else {
          throw new Error(
            "Document not found. Please upload the document first or it was cleared."
          );
        }
      }

      logger.info("Processing questions", {
        questionsCount: questions.length,
        storeId,
        requestId,
      });

      const answers = [];

      // Process questions in parallel for speed
      const answerPromises = questions.map(async (question, index) => {
        const questionId = `q${index + 1}`;
        logger.info("Processing question", {
          questionId,
          question: question.substring(0, 100),
          requestId,
        });

        // Check cache for this specific question + document combination
        const questionCacheKey = `answer:${storeId}:${Buffer.from(
          question
        ).toString("base64")}`;
        let cachedAnswer;
        try {
          cachedAnswer = await redisClient.get(questionCacheKey);
          if (cachedAnswer) {
            logger.info("Found cached answer", { questionId, requestId });
            return { index, answer: JSON.parse(cachedAnswer) };
          }
        } catch (redisError) {
          logger.warn("Answer cache check failed", {
            error: redisError.message,
            requestId,
          });
        }

        // Retrieve only top 10 most relevant docs
        const relevantDocs = await vectorStore.similaritySearch(question, 10);

        // Prepare context from relevant documents
        const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

        // Optimized prompt for faster response
        const prompt = `You are an expert assistant. Use ONLY the following context to answer the question.

Context:
${context}

Question:
${question}

Instructions:
- Give a one-sentence answer that includes all critical details from the context.
- Limit to 35 words max.
- Do NOT add any explanation or repeat the question.
- If the answer is not in the context, try harder and infer from the document. You cannot say document not found unless you are 200% sure its not there and you cant infer from rest of document
- Respond only with a JSON object in this format: {"answer": "your answer here"}

Answer:`;

        try {
          const chatCompletion = await groq.chat.completions.create({
            messages: [
              {
                role: "user",
                content: prompt,
              },
            ],
            model: "gemma2-9b-it",
            temperature: 0.1, // Lower temperature for more consistent answers
            max_completion_tokens: 1024,
            top_p: 1,
            stream: false,
            response_format: {
              type: "json_object",
            },
            stop: null,
          });

          const responseContent = chatCompletion.choices[0].message.content;
          let parsedAnswer;

          try {
            const jsonResponse = JSON.parse(responseContent);
            parsedAnswer = [jsonResponse.answer || responseContent];
          } catch (parseError) {
            // Fallback if JSON parsing fails
            parsedAnswer = [responseContent];
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
            logger.warn("Failed to cache answer", {
              error: redisError.message,
              requestId,
            });
          }

          logger.info("Question processed", {
            questionId,
            answerLength: parsedAnswer[0]?.length || 0,
            requestId,
          });

          return { index, answer: parsedAnswer };
        } catch (groqError) {
          logger.error("Groq API error", {
            questionId,
            error: groqError.message,
            requestId,
          });
          // Fallback answer
          return {
            index,
            answer: ["Unable to process question due to API error."],
          };
        }
      });

      // Wait for all answers to complete
      const results = await Promise.all(answerPromises);

      // Sort by original order
      results.sort((a, b) => a.index - b.index);

      logger.info("All questions processed", {
        questionsCount: questions.length,
        storeId,
        requestId,
      });

      return results.flatMap((r) => r.answer);
    } catch (error) {
      logger.error("Error answering questions", {
        storeId,
        requestId,
        error: error.message,
        stack: error.stack,
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

    logger.info("File upload started", {
      filename: req.file.originalname,
      size: req.file.size,
      mimeType: req.file.mimetype,
      requestId: req.requestId,
    });

    const result = await documentProcessor.processAndStoreDocument(
      req.file.path,
      req.requestId,
      req.file.mimetype // Pass file type explicitly for file uploads
    );

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    logger.info("File upload completed", {
      storeId: result.storeId,
      chunksCount: result.chunksCount,
      requestId: req.requestId,
    });

    res.json({
      message: "Document processed successfully",
      storeId: result.storeId, // Now the content hash
      chunksCount: result.chunksCount,
    });
  } catch (error) {
    logger.error("Upload error", {
      error: error.message,
      requestId: req.requestId,
      stack: error.stack,
    });
    res.status(500).json({ error: error.message });
  }
});

// Main endpoint matching the sample request format
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();
  const requestId = req.requestId;

  // Create a dedicated HackRX logger for this request
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

  try {
    // Validate authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      logger.warn("Invalid authorization header", { requestId });
      hackrxLogger.warn("Invalid authorization header", { requestId });
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
        requestId,
      });
      hackrxLogger.warn("Invalid request format", {
        hasDocuments: !!documents,
        hasQuestions: !!questions,
        questionsIsArray: Array.isArray(questions),
        requestId,
      });
      return res.status(400).json({
        error:
          "Invalid request format. Expected documents (string) and questions (array)",
      });
    }

    logger.info("HackRX Request started", {
      questionsCount: questions.length,
      documentInputLength: documents.length,
      requestId,
      authHeader: authHeader.substring(0, 20) + "...",
    });

    // Log the full request details to HackRX log file
    hackrxLogger.info("HackRX Request Details", {
      requestId,
      timestamp: new Date().toISOString(),
      headers: req.headers,
      body: {
        documents: documents.substring(0, 500) + "...", // Truncate for log
        questions: questions,
      },
    });

    // Process the document - this will return the contentHash as storeId
    logger.info("Processing document source", {
      documentSourcePreview: documents.substring(0, 100) + "...",
      requestId,
    });

    const docProcessingResult = await documentProcessor.processAndStoreDocument(
      documents,
      requestId
    );
    const storeId = docProcessingResult.storeId; // This is now the content hash

    logger.info("Document processed (or loaded from cache)", {
      storeId,
      chunksCount: docProcessingResult.chunksCount,
      requestId,
    });
    hackrxLogger.info("Document processed (or loaded from cache)", {
      storeId,
      chunksCount: docProcessingResult.chunksCount,
      requestId,
    });

    // Answer the questions
    logger.info("Answering questions", {
      questionsCount: questions.length,
      storeId,
      requestId,
    });

    const answers = await documentProcessor.answerQuestions(
      questions,
      storeId, // Pass the content hash as storeId
      requestId
    );

    const processingTime = Date.now() - startTime;

    logger.info("HackRX Request completed", {
      processingTime: `${processingTime}ms`,
      answersCount: answers.length,
      storeId,
      requestId,
    });

    // Log the response to HackRX log file
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
      requestId,
    });
    hackrxLogger.error("HackRX error", {
      processingTime: `${processingTime}ms`,
      error: error.message,
      stack: error.stack,
      requestId,
    });

    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  } finally {
    // Ensure the hackrxLogger is closed/flushed
    hackrxLogger.end();
  }
});

// Answer questions for existing document
app.post("/answer/:storeId", async (req, res) => {
  try {
    const { storeId } = req.params; // storeId is now expected to be the content hash
    const { questions } = req.body;

    if (!questions || !Array.isArray(questions)) {
      return res
        .status(400)
        .json({ error: "Questions must be provided as an array" });
    }

    logger.info("Answering questions for existing document", {
      storeId,
      questionsCount: questions.length,
      requestId: req.requestId,
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
      requestId: req.requestId,
    });
    res.status(500).json({ error: error.message });
  }
});

// Get available document stores (lists FAISS files by their content hash names)
app.get("/stores", (req, res) => {
  try {
    const stores = fs
      .readdirSync(faissStoresDir)
      .filter((file) => file.endsWith(".faiss"))
      .map((file) => file.replace(".faiss", "")); // Just the hash
    logger.info("Document stores requested", {
      storesCount: stores.length,
      requestId: req.requestId,
    });
    res.json({ stores: stores, count: stores.length });
  } catch (error) {
    logger.error("Error listing stores", {
      error: error.message,
      requestId: req.requestId,
    });
    res.status(500).json({ error: "Failed to list stores" });
  }
});

// Delete a document store (FAISS file and in-memory if present)
app.delete("/stores/:storeId", (req, res) => {
  const { storeId } = req.params; // storeId is the content hash

  const faissStorePath = path.join(faissStoresDir, `${storeId}.faiss`);

  if (fs.existsSync(faissStorePath)) {
    try {
      fs.unlinkSync(faissStorePath);
      vectorStores.delete(storeId); // Also remove from in-memory cache
      logger.info("Document store deleted", {
        storeId,
        faissStorePath,
        requestId: req.requestId,
      });
      res.json({ message: "Document store deleted successfully" });
    } catch (error) {
      logger.error("Error deleting document store file", {
        storeId,
        faissStorePath,
        error: error.message,
        requestId: req.requestId,
      });
      res.status(500).json({ error: "Failed to delete document store" });
    }
  } else {
    logger.warn("Document store not found for deletion", {
      storeId,
      requestId: req.requestId,
    });
    res.status(404).json({ error: "Document store not found" });
  }
});

// Clear cache endpoint - now also removes FAISS files
app.post("/cache/clear", async (req, res) => {
  try {
    await redisClient.flushAll(); // Clear Redis cache
    logger.info("Redis cache cleared", { requestId: req.requestId });

    // Clear FAISS store files
    fs.readdirSync(faissStoresDir).forEach((file) => {
      if (file.endsWith(".faiss")) {
        fs.unlinkSync(path.join(faissStoresDir, file));
      }
    });
    vectorStores.clear(); // Clear in-memory map
    logger.info("FAISS stores cleared from disk and memory", {
      requestId: req.requestId,
    });

    res.json({ message: "All caches cleared successfully" });
  } catch (error) {
    logger.error("Error clearing cache", {
      error: error.message,
      requestId: req.requestId,
    });
    res.status(500).json({ error: "Failed to clear cache" });
  }
});

// Get cache stats
app.get("/cache/stats", async (req, res) => {
  try {
    const info = await redisClient.info();
    const keyCount = await redisClient.dbSize();
    const faissFiles = fs
      .readdirSync(faissStoresDir)
      .filter((file) => file.endsWith(".faiss")).length;

    res.json({
      connected: true,
      redisKeyCount: keyCount,
      faissStoresOnDisk: faissFiles,
      inMemoryFaissStores: vectorStores.size,
      info: info.split("\r\n").reduce((acc, line) => {
        const [key, value] = line.split(":");
        if (key && value) acc[key] = value;
        return acc;
      }, {}),
    });
  } catch (error) {
    logger.error("Error getting cache stats", {
      error: error.message,
      requestId: req.requestId,
    });
    res.status(500).json({
      connected: false,
      error: error.message,
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error("Unhandled error", {
    error: error.message,
    stack: error.stack,
    requestId: req.requestId,
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
    requestId: req.requestId,
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
  console.log(
    "  POST /answer/:storeId - Answer questions for existing document"
  );
  console.log("  GET /stores - List document stores");
  console.log("  DELETE /stores/:storeId - Delete document store");
  console.log("  POST /cache/clear - Clear Redis cache and FAISS files");
  console.log("  GET /cache/stats - Get cache statistics");
  console.log("  GET /health - Health check");
  console.log(`Logs are being saved to: ${logsDir}`);
  console.log(`FAISS stores are being saved to: ${faissStoresDir}`);
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
  logger.error("Uncaught Exception", {
    error: error.message,
    stack: error.stack,
  });
  // Consider more robust error handling / process restart strategies in production
  process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection", { reason, promise });
});

module.exports = app;
