const express = require("express");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { Groq } = require("groq-sdk");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const redis = require("redis");
const crypto = require("crypto");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware - minimal setup
app.use(cors());
app.use(express.json({ limit: "1000mb" }));

// Initialize clients
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004",
});

// Redis client for ultra-fast caching
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});
redisClient.connect().catch(console.error);

// In-memory stores for maximum speed
const vectorStores = new Map();
const textCache = new Map();
const answerCache = new Map();

// Ultra-optimized document processor
class UltraFastProcessor {
  constructor() {
    // Aggressive chunking for speed - larger chunks, less overlap
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000, // Larger chunks = fewer embeddings
      chunkOverlap: 100, // Minimal overlap
    });
  }

  async downloadPDF(url) {
    const urlHash = crypto.createHash("md5").update(url).digest("hex");

    // Check in-memory cache first
    if (textCache.has(urlHash)) {
      return { text: textCache.get(urlHash), fromCache: true };
    }

    // Check Redis cache
    try {
      const cached = await redisClient.get(`pdf:${urlHash}`);
      if (cached) {
        textCache.set(urlHash, cached); // Also store in memory
        return { text: cached, fromCache: true };
      }
    } catch (e) {
      console.warn("Redis cache miss:", e.message);
    }

    // Download with aggressive settings
    console.log(`⏳ Downloading PDF: ${url} `);
    const startTime = Date.now();

    const response = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 60000, // 60s max for download (increased)
      maxContentLength: 500 * 1024 * 1024, // 500MB (increased)
      maxBodyLength: 500 * 1024 * 1024, // 500MB (increased)
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      },
    });

    console.log(
      `✅ Downloaded in ${Date.now() - startTime}ms, size: ${(
        response.data.length /
        1024 /
        1024
      ).toFixed(2)}MB`
    );

    // Parse PDF with optimized settings
    const parseStart = Date.now();
    const data = await pdfParse(response.data, {
      max: 0, // No limit on pages
      version: "v1.10.100",
    });

    console.log(
      `✅ Parsed PDF in ${Date.now() - parseStart}ms, ${data.numpages} pages, ${
        data.text.length
      } chars`
    );

    // Cache aggressively
    textCache.set(urlHash, data.text);
    try {
      await redisClient.setEx(`pdf:${urlHash}`, 3600, data.text); // 1 hour cache
    } catch (e) {
      console.warn("Redis cache write failed:", e.message);
    }

    return { text: data.text, fromCache: false };
  }

  async createVectorStore(text, contentHash) {
    // Check if vector store exists in memory
    if (vectorStores.has(contentHash)) {
      return { store: vectorStores.get(contentHash), fromCache: true };
    }

    console.log(
      `⏳ Creating vector store for ${(text.length / 1000).toFixed(
        1
      )}k chars...`
    );
    const startTime = Date.now();

    // Create chunks
    const docs = await this.textSplitter.createDocuments([text]);
    console.log(
      `✅ Created ${docs.length} chunks in ${Date.now() - startTime}ms`
    );

    // Create embeddings in parallel batches for speed
    const embeddingStart = Date.now();
    const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
    console.log(`✅ Vector store created in ${Date.now() - embeddingStart}ms`);

    // Cache in memory for instant future access
    vectorStores.set(contentHash, vectorStore);

    return { store: vectorStore, fromCache: false };
  }

  async answerQuestions(questions, vectorStore, contentHash) {
    console.log(`⏳ Processing ${questions.length} questions...`);
    const startTime = Date.now();

    // Process questions in parallel with limited concurrency to avoid rate limits
    const batchSize = 5; // Process 5 questions at a time
    const results = [];

    for (let i = 0; i < questions.length; i += batchSize) {
      const batch = questions.slice(i, i + batchSize);
      const batchPromises = batch.map(async (question, batchIndex) => {
        const questionIndex = i + batchIndex;

        // Check answer cache
        const cacheKey = `${contentHash}:${crypto
          .createHash("md5")
          .update(question)
          .digest("hex")}`;
        if (answerCache.has(cacheKey)) {
          return { index: questionIndex, answer: answerCache.get(cacheKey) };
        }

        try {
          // Get only top 3 most relevant chunks for speed
          const docs = await vectorStore.similaritySearch(question, 3);
          const context = docs.map((doc) => doc.pageContent).join("\n\n");

          // Ultra-concise prompt for fastest response
          const prompt = `Context: ${context}\n\nQuestion: ${question}\n\nAnswer in max 25 words as JSON: {"answer": "..."}`;

          const response = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant", // Fastest model
            temperature: 0,
            max_completion_tokens: 100,
            response_format: { type: "json_object" },
          });

          let answer;
          try {
            const parsed = JSON.parse(response.choices[0].message.content);
            answer = parsed.answer || "No answer found";
          } catch {
            answer = response.choices[0].message.content || "Parse error";
          }

          // Cache the answer
          answerCache.set(cacheKey, answer);

          return { index: questionIndex, answer };
        } catch (error) {
          console.error(
            `Error processing question ${questionIndex}:`,
            error.message
          );
          return { index: questionIndex, answer: "Processing error" };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    }

    // Sort by original order
    results.sort((a, b) => a.index - b.index);
    const answers = results.map((r) => r.answer);

    console.log(`✅ All questions answered in ${Date.now() - startTime}ms`);
    return answers;
  }
}

const processor = new UltraFastProcessor();

// Hardcoded bearer token for validation
const VALID_BEARER_TOKEN = process.env.TOKEN;

// SINGLE OPTIMIZED ENDPOINT
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}`;

  try {
    // Check Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      console.warn(`❌ [${requestId}] Missing or invalid Authorization header`);
      return res.status(401).json({
        error:
          "Missing or invalid Authorization header. Expected 'Bearer <token>'",
      });
    }

    // Extract and validate the bearer token
    const token = authHeader.substring(7); // Remove "Bearer " prefix
    if (token !== VALID_BEARER_TOKEN) {
      console.warn(
        `❌ [${requestId}] Invalid bearer token: ${token.substring(0, 10)}...`
      );
      return res.status(401).json({
        error: "Invalid bearer token",
      });
    }

    // Check Content-Type header
    const contentType = req.headers["content-type"];
    if (!contentType || !contentType.includes("application/json")) {
      console.warn(`❌ [${requestId}] Invalid Content-Type header`);
      return res.status(400).json({
        error: "Invalid Content-Type header. Expected 'application/json'",
      });
    }

    console.log(`✅ [${requestId}] Headers validated successfully`);

    const { documents, questions } = req.body;

    // Basic validation
    if (!documents || !questions || !Array.isArray(questions)) {
      return res.status(400).json({
        error:
          "Invalid request format. Expected documents (string) and questions (array)",
      });
    }

    console.log(
      `🚀 [${requestId}] Processing ${questions.length} questions for document`
    );

    // Step 1: Download and parse PDF (target: <10s)
    const { text, fromCache: textCached } = await processor.downloadPDF(
      documents
    );
    const downloadTime = Date.now() - startTime;
    console.log(
      `📄 Text ${
        textCached ? "loaded from cache" : "downloaded"
      } in ${downloadTime}ms`
    );

    // Step 2: Create vector store (target: <15s total)
    const contentHash = crypto.createHash("md5").update(text).digest("hex");
    const { store, fromCache: storeCached } = await processor.createVectorStore(
      text,
      contentHash
    );
    const vectorTime = Date.now() - startTime;
    console.log(
      `🔍 Vector store ${
        storeCached ? "loaded from cache" : "created"
      } in ${vectorTime}ms`
    );

    // Step 3: Answer questions (target: <25s total)
    const answers = await processor.answerQuestions(
      questions,
      store,
      contentHash
    );
    const totalTime = Date.now() - startTime;

    console.log(
      `✅ [${requestId}] Completed in ${totalTime}ms (${(
        totalTime / 1000
      ).toFixed(1)}s)`
    );

    // Performance warning if over 25s
    if (totalTime > 25000) {
      console.warn(
        `⚠️  Response time ${(totalTime / 1000).toFixed(1)}s exceeds 25s target`
      );
    }

    res.json({
      answers: answers,
      // Optional debug info (remove in production)
    });
  } catch (error) {
    const errorTime = Date.now() - startTime;
    console.error(
      `❌ [${requestId}] Error after ${errorTime}ms:`,
      error.message
    );

    res.status(500).json({
      error: "Processing failed",
      message: error.message,
      processingTime: `${errorTime}ms`,
    });
  }
});

// Minimal health check
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    caches: {
      textCache: textCache.size,
      vectorStores: vectorStores.size,
      answerCache: answerCache.size,
    },
  });
});

// Cache management
app.post("/cache/clear", (req, res) => {
  textCache.clear();
  vectorStores.clear();
  answerCache.clear();
  redisClient.flushAll().catch(console.error);
  res.json({ message: "All caches cleared" });
});

// Start server with optimized settings
const server = app.listen(port, () => {
  console.log(`🚀 Ultra-Fast PDF Server running on port ${port}`);
  console.log(`📊 Target: <30s response time for 1000+ page PDFs`);
  console.log(
    `🔧 Optimizations: Aggressive caching, parallel processing, minimal chunks`
  );
});

server.setTimeout(35000); // 35s timeout
server.keepAliveTimeout = 30000;
server.headersTimeout = 31000;

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\n🛑 Shutting down...");
  try {
    await redisClient.quit();
  } catch (e) {
    console.error("Redis cleanup error:", e.message);
  }
  process.exit(0);
});

module.exports = app;
