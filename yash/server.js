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
app.use(express.json({ limit: "100mb" }));

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
        textCache.set(urlHash, cached);
        return { text: cached, fromCache: true };
      }
    } catch (e) {
      console.warn("Redis cache miss:", e.message);
    }

    console.log(`⏳ Downloading PDF: ${url.substring(0, 50)}...`);
    const startTime = Date.now();

    // Multi-threaded download with aggressive settings
    const response = await axios.get(url, {
      responseType: "stream", // Stream instead of loading into memory
      timeout: 20000, // Reduced to 20s max
      maxContentLength: 200 * 1024 * 1024,
      maxBodyLength: 200 * 1024 * 1024,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        Accept: "application/pdf,application/octet-stream,*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
        "Cache-Control": "no-cache",
      },
      // AWS-specific optimizations
      httpAgent: new (require("http").Agent)({
        keepAlive: true,
        maxSockets: 50,
        timeout: 20000,
      }),
      httpsAgent: new (require("https").Agent)({
        keepAlive: true,
        maxSockets: 50,
        timeout: 20000,
        rejectUnauthorized: false, // For speed, but risky
      }),
    });

    // Stream to buffer for faster processing
    const chunks = [];
    let totalLength = 0;

    response.data.on("data", (chunk) => {
      chunks.push(chunk);
      totalLength += chunk.length;

      // Progress logging every 1MB
      if (totalLength % (1024 * 1024) === 0) {
        console.log(
          `📥 Downloaded ${(totalLength / 1024 / 1024).toFixed(1)}MB...`
        );
      }
    });

    const buffer = await new Promise((resolve, reject) => {
      response.data.on("end", () => {
        resolve(Buffer.concat(chunks, totalLength));
      });
      response.data.on("error", reject);
    });

    console.log(
      `✅ Downloaded in ${Date.now() - startTime}ms, size: ${(
        buffer.length /
        1024 /
        1024
      ).toFixed(2)}MB`
    );

    // Parse PDF with streaming
    const parseStart = Date.now();
    const data = await pdfParse(buffer, {
      max: 0,
      version: "v1.10.100",
      // Optimize parsing
      normalizeWhitespace: false,
      disableCombineTextItems: true,
    });

    console.log(
      `✅ Parsed PDF in ${Date.now() - parseStart}ms, ${data.numpages} pages`
    );

    // Cache aggressively
    textCache.set(urlHash, data.text);
    try {
      await redisClient.setEx(`pdf:${urlHash}`, 7200, data.text); // 2 hour cache
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

          // More concise prompt with shorter context
          const prompt = `Context: ${context.substring(0, 800)}

Question: ${question}

Answer in 1-2 sentences. JSON format: {"answer": "brief answer"}`;

          const response = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant", // Fastest model
            temperature: 0,
            max_completion_tokens: 150,
            response_format: { type: "json_object" },
          });

          let answer;
          try {
            const parsed = JSON.parse(response.choices[0].message.content);
            answer = parsed.answer || "No answer found";
          } catch (parseError) {
            // Better fallback parsing
            const rawContent = response.choices[0].message.content || "";
            console.warn(
              `JSON parse failed for question ${questionIndex}, raw: ${rawContent.substring(
                0,
                100
              )}`
            );

            // Try to extract answer from malformed JSON
            const answerMatch = rawContent.match(/"answer"\s*:\s*"([^"]+)"/);
            if (answerMatch) {
              answer = answerMatch[1];
            } else {
              // Last resort: use first sentence
              answer =
                rawContent.split(".")[0].substring(0, 100) || "Parse error";
            }
          }

          // Cache the answer
          answerCache.set(cacheKey, answer);

          return { index: questionIndex, answer };
        } catch (error) {
          console.error(
            `Error processing question ${questionIndex}:`,
            error.message
          );

          // Better fallback without JSON formatting
          try {
            const docs = await vectorStore.similaritySearch(question, 2);
            const context = docs.map((doc) => doc.pageContent).join("\n");

            const fallbackResponse = await groq.chat.completions.create({
              messages: [
                {
                  role: "user",
                  content: `Context: ${context.substring(0, 600)}

Question: ${question}

Answer in one coherent sentence as JSON: {"answer": "..."}:`, // Reverting to JSON format even for fallback
                },
              ],
              model: "llama-3.1-8b-instant",
              temperature: 0,
              response_format: { type: "json_object" }, // Ensure JSON expected
              max_completion_tokens: 100,
            });

            let fallbackAnswer;
            try {
              const parsed = JSON.parse(
                fallbackResponse.choices[0].message.content
              );
              fallbackAnswer = parsed.answer || "No fallback answer found";
            } catch (fallbackParseError) {
              const rawContent =
                fallbackResponse.choices[0].message.content || "";
              console.warn(
                `Fallback JSON parse failed for question ${questionIndex}, raw: ${rawContent.substring(
                  0,
                  100
                )}`
              );
              const answerMatch = rawContent.match(/"answer"\s*:\s*"([^"]+)"/);
              if (answerMatch) {
                fallbackAnswer = answerMatch[1];
              } else {
                fallbackAnswer =
                  rawContent.split(".")[0].substring(0, 100) ||
                  "Fallback parse error";
              }
            }

            const cleanAnswer = fallbackAnswer || "Unable to process question";
            answerCache.set(cacheKey, cleanAnswer);
            return { index: questionIndex, answer: cleanAnswer };
          } catch (fallbackError) {
            console.error(
              `Fallback also failed for question ${questionIndex}:`,
              fallbackError.message
            );
            return {
              index: questionIndex,
              answer: "Unable to process question",
            };
          }
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      // ADDED: Small delay between batches to avoid rate limiting
      if (i + batchSize < questions.length) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }

    // Sort by original order
    results.sort((a, b) => a.index - b.index);
    const answers = results.map((r) => r.answer);

    console.log(`✅ All questions answered in ${Date.now() - startTime}ms`);
    return answers;
  }
}

const processor = new UltraFastProcessor();

// SINGLE OPTIMIZED ENDPOINT (Non-Streaming)
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}`;

  try {
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
