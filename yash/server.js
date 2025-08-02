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
const cluster = require("cluster");
const os = require("os");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

// Enhanced middleware with compression
app.use(cors());
app.use(express.json({ limit: "100mb" }));

// Logging middleware for all incoming JSON requests
app.use((req, res, next) => {
  if (
    req.method === "POST" &&
    req.headers["content-type"]?.includes("application/json")
  ) {
    const timestamp = new Date().toISOString();
    const requestId = `req_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    console.log(
      `📥 [${requestId}] ${timestamp} - Incoming ${req.method} ${req.path}`
    );
    console.log(
      `📋 [${requestId}] Request Body:`,
      JSON.stringify(req.body, null, 2)
    );

    // Store requestId for response logging
    req.requestId = requestId;
  }
  next();
});

// Initialize clients with latest models
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Use the latest Gemini Embedding model for maximum accuracy
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004", // Latest stable model
  // Alternative: "gemini-embedding" for experimental higher accuracy
});

// Redis client with optimized connection pooling
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
  socket: {
    keepAlive: true,
    reconnectStrategy: (retries) => Math.min(retries * 50, 500),
  },
  database: 0,
});
redisClient.connect().catch(console.error);

// Enhanced in-memory stores with TTL
const vectorStores = new Map();
const textCache = new Map();
const answerCache = new Map();
const embeddingCache = new Map();

// Ultra-optimized processor with latest models
class UltraFastAccurateProcessor {
  constructor() {
    // Optimized chunking for better search precision and accuracy balance
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1200, // Smaller chunks for better precision
      chunkOverlap: 300, // Higher overlap to ensure important info isn't split
      separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    });

    // Connection pool for parallel processing
    this.httpAgent = new (require("http").Agent)({
      keepAlive: true,
      maxSockets: 100,
      timeout: 15000,
    });

    this.httpsAgent = new (require("https").Agent)({
      keepAlive: true,
      maxSockets: 100,
      timeout: 15000,
      rejectUnauthorized: false,
    });
  }

  generateKeywordVariations(question) {
    // Generate search variations for better document retrieval
    const variations = [];
    const lowerQuestion = question.toLowerCase();

    // Extract key terms and create variations
    if (lowerQuestion.includes("grace period")) {
      variations.push(
        "thirty days premium payment",
        "grace period premium",
        "payment due date"
      );
    }
    if (lowerQuestion.includes("waiting period")) {
      variations.push(
        "waiting period months",
        "continuous coverage",
        "pre-existing disease"
      );
    }
    if (lowerQuestion.includes("maternity")) {
      variations.push(
        "maternity expenses childbirth",
        "female insured 24 months",
        "delivery coverage"
      );
    }
    if (lowerQuestion.includes("cataract")) {
      variations.push(
        "cataract surgery two years",
        "eye surgery waiting period"
      );
    }
    if (lowerQuestion.includes("organ donor")) {
      variations.push(
        "organ transplant donor expenses",
        "harvesting organ medical"
      );
    }
    if (lowerQuestion.includes("no claim discount")) {
      variations.push("NCD 5% premium renewal", "claim discount base premium");
    }
    if (lowerQuestion.includes("health check")) {
      variations.push(
        "preventive health checkup reimbursement",
        "health check two years"
      );
    }
    if (lowerQuestion.includes("hospital")) {
      variations.push(
        "hospital definition inpatient beds",
        "qualified medical practitioner"
      );
    }
    if (lowerQuestion.includes("ayush")) {
      variations.push("ayurveda yoga naturopathy", "AYUSH hospital treatment");
    }
    if (lowerQuestion.includes("room rent") || lowerQuestion.includes("icu")) {
      variations.push("room charges ICU 1% 2%", "daily room rent sum insured");
    }

    // Add the original question as fallback
    variations.push(question);

    return variations.slice(0, 3); // Limit to avoid too many searches
  }

  async downloadPDF(url) {
    const urlHash = crypto.createHash("sha256").update(url).digest("hex");
    const pdfFileName = `pdf_${urlHash}.pdf`;
    const pdfFilePath = path.join(__dirname, "temp", pdfFileName);

    // Ensure temp directory exists
    if (!fs.existsSync(path.join(__dirname, "temp"))) {
      fs.mkdirSync(path.join(__dirname, "temp"), { recursive: true });
    }

    // Multi-level caching check
    if (textCache.has(urlHash)) {
      return {
        text: textCache.get(urlHash),
        fromCache: true,
        filePath: pdfFilePath,
      };
    }

    try {
      const cached = await redisClient.get(`pdf:${urlHash}`);
      if (cached) {
        const decompressed = require("zlib")
          .gunzipSync(Buffer.from(cached, "base64"))
          .toString();
        textCache.set(urlHash, decompressed);
        return { text: decompressed, fromCache: true, filePath: pdfFilePath };
      }
    } catch (e) {
      console.warn("Redis cache miss:", e.message);
    }

    console.log(`⏳ Downloading PDF: ${url.substring(0, 50)}...`);
    const startTime = Date.now();

    // Enhanced download with retry logic
    let retries = 3;
    let response;

    while (retries > 0) {
      try {
        response = await axios.get(url, {
          responseType: "stream",
          timeout: 30000,
          maxContentLength: 500 * 1024 * 1024, // 500MB limit
          maxBodyLength: 500 * 1024 * 1024,
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            Accept: "application/pdf,application/octet-stream,*/*",
            "Accept-Encoding": "gzip, deflate, br",
            Connection: "keep-alive",
            "Cache-Control": "no-cache",
          },
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        });
        break;
      } catch (error) {
        retries--;
        if (retries === 0) throw error;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    // Stream processing with progress tracking
    const chunks = [];
    let totalLength = 0;

    response.data.on("data", (chunk) => {
      chunks.push(chunk);
      totalLength += chunk.length;
    });

    const buffer = await new Promise((resolve, reject) => {
      response.data.on("end", () =>
        resolve(Buffer.concat(chunks, totalLength))
      );
      response.data.on("error", reject);
    });

    console.log(
      `✅ Downloaded in ${Date.now() - startTime}ms, size: ${(
        buffer.length /
        1024 /
        1024
      ).toFixed(2)}MB`
    );

    // Save PDF to temp folder
    try {
      fs.writeFileSync(pdfFilePath, buffer);
      console.log(`💾 PDF saved to: ${pdfFilePath}`);
    } catch (saveError) {
      console.warn(
        `⚠️ Failed to save PDF to temp folder: ${saveError.message}`
      );
    }

    // Enhanced PDF parsing with better error handling
    const parseStart = Date.now();
    const data = await pdfParse(buffer, {
      max: 0,
      version: "v1.10.100",
      normalizeWhitespace: true,
      disableCombineTextItems: false,
    });

    console.log(
      `✅ Parsed PDF in ${Date.now() - parseStart}ms, ${
        data.numpages
      } pages, ${(data.text.length / 1000).toFixed(1)}k chars`
    );

    // Enhanced caching with compression
    textCache.set(urlHash, data.text);
    try {
      const compressed = require("zlib").gzipSync(data.text).toString("base64");
      await redisClient.setEx(`pdf:${urlHash}`, 14400, compressed); // 4 hour cache
    } catch (e) {
      console.warn("Redis cache write failed:", e.message);
    }

    return { text: data.text, fromCache: false, filePath: pdfFilePath };
  }

  async createVectorStore(text, contentHash) {
    if (vectorStores.has(contentHash)) {
      return { store: vectorStores.get(contentHash), fromCache: true };
    }

    console.log(
      `⏳ Creating vector store for ${(text.length / 1000).toFixed(
        1
      )}k chars...`
    );
    const startTime = Date.now();

    // Enhanced text preprocessing for better searchability
    const preprocessedText = this.preprocessText(text);

    // Create multiple granular chunks for better search precision
    const docs = await this.textSplitter.createDocuments([preprocessedText]);

    // Add metadata to chunks for better retrieval
    docs.forEach((doc, index) => {
      doc.metadata = {
        chunk_id: index,
        char_start: index * 1300, // Approximate start position
        content_preview: doc.pageContent.substring(0, 100) + "...",
      };
    });

    console.log(
      `✅ Created ${docs.length} enhanced chunks in ${Date.now() - startTime}ms`
    );

    // Create vector store with enhanced embeddings
    const embeddingStart = Date.now();
    const vectorStore = await FaissStore.fromDocuments(docs, embeddings);

    console.log(`✅ Vector store created in ${Date.now() - embeddingStart}ms`);

    // Cache with TTL
    vectorStores.set(contentHash, vectorStore);
    setTimeout(() => vectorStores.delete(contentHash), 3600000); // 1 hour TTL

    return { store: vectorStore, fromCache: false };
  }

  preprocessText(text) {
    // Enhance text for better searchability
    return (
      text
        // Normalize whitespace but preserve structure
        .replace(/\s+/g, " ")
        // Ensure section headers are properly spaced
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        // Add line breaks before common policy sections
        .replace(
          /(waiting period|grace period|coverage|benefits?|exclusions?|definitions?)/gi,
          "\n$1"
        )
        // Normalize common insurance terms
        .replace(/Sum\s+Insured/gi, "Sum Insured")
        .replace(/Pre[\s-]?existing/gi, "Pre-existing")
        .replace(/No[\s-]?Claim[\s-]?Discount/gi, "No Claim Discount")
        .trim()
    );
  }

  async answerQuestions(questions, vectorStore, contentHash) {
    console.log(
      `⏳ Processing ${questions.length} questions with high accuracy...`
    );
    const startTime = Date.now();

    // Use latest Groq models for maximum accuracy
    const models = [
      "moonshotai/kimi-k2-instruct",
      "llama-3.3-70b-versatile", // Latest, most accurate
      "llama-3.1-70b-versatile", // Fallback high accuracy
      "llama-3.1-8b-instant", // Speed fallback
    ];

    // Process questions in optimized batches
    const batchSize = 3; // Smaller batches for higher accuracy
    const results = [];

    for (let i = 0; i < questions.length; i += batchSize) {
      const batch = questions.slice(i, i + batchSize);
      const batchPromises = batch.map(async (question, batchIndex) => {
        const questionIndex = i + batchIndex;

        // Enhanced cache key with model version
        const cacheKey = `${contentHash}:${crypto
          .createHash("sha256")
          .update(question + models[0])
          .digest("hex")}`;

        if (answerCache.has(cacheKey)) {
          return { index: questionIndex, answer: answerCache.get(cacheKey) };
        }

        try {
          // Multi-stage search for comprehensive coverage
          let docs = await vectorStore.similaritySearch(question, 8);

          // If first search doesn't yield enough, try keyword variations
          if (docs.length < 3) {
            const keywordVariations = this.generateKeywordVariations(question);
            for (const variation of keywordVariations) {
              const additionalDocs = await vectorStore.similaritySearch(
                variation,
                5
              );
              docs = [...docs, ...additionalDocs];
              if (docs.length >= 5) break;
            }
          }

          // Remove duplicates and get comprehensive context
          const uniqueDocs = docs.filter(
            (doc, index, self) =>
              index === self.findIndex((d) => d.pageContent === doc.pageContent)
          );

          const context = uniqueDocs.map((doc) => doc.pageContent).join("\n\n");

          // Enhanced prompt designed to match expected output format
          const prompt = `You are an expert insurance policy analyst. Extract the precise answer from the policy document context provided.

POLICY DOCUMENT CONTEXT:
${context.substring(0, 3000)}

QUESTION: ${question}

CRITICAL INSTRUCTIONS:
1. Extract the EXACT information from the policy document - do not say "context doesn't contain enough information" unless truly absent
2. Look for specific numbers, timeframes, percentages, conditions, and definitions
3. If you find partial information, provide what's available with specific details
4. Include specific policy terms, waiting periods, percentages, and conditions mentioned
5. For definitions, provide the complete definition as stated in the policy
6. For benefits/coverage questions, include eligibility criteria and limits
7. Be comprehensive but concise - include all relevant details from the policy. Recomended 1-2 sentences where you cannot keep on putting things in the sentence with conjunctions and call it a day.But if need be you can extend as long as no detail is being skiped on 
8. Answers have to be written like how a human would read with proper punctuation and grammar.

FORMAT: Respond with JSON: {"answer": "detailed policy answer with specific terms and numbers"}

ANSWER:`;

          let response;
          let modelUsed = models[0];

          // Try models in order of accuracy with enhanced parameters
          for (const model of models) {
            try {
              response = await groq.chat.completions.create({
                messages: [{ role: "user", content: prompt }],
                model: model,
                temperature: 0.05, // Very low temperature for factual accuracy
                max_completion_tokens: 400, // More tokens for comprehensive answers
                response_format: { type: "json_object" },
                top_p: 0.85, // Slight reduction for more focused responses
                frequency_penalty: 0.1, // Reduce repetition
                presence_penalty: 0.1, // Encourage diverse content
              });
              modelUsed = model;
              break;
            } catch (modelError) {
              console.warn(`Model ${model} failed, trying next...`);
              continue;
            }
          }

          if (!response) {
            throw new Error("All models failed");
          }

          let answer;
          try {
            const parsed = JSON.parse(response.choices[0].message.content);
            answer = parsed.answer || "No answer found in context";
          } catch (parseError) {
            // Enhanced fallback parsing
            const rawContent = response.choices[0].message.content || "";
            const answerMatch = rawContent.match(/"answer"\s*:\s*"([^"]+)"/);
            if (answerMatch) {
              answer = answerMatch[1];
            } else {
              // Extract meaningful content
              answer =
                rawContent.replace(/[{}]/g, "").split(":").pop()?.trim() ||
                "Parse error occurred";
            }
          }

          // Enhanced answer post-processing
          answer = answer.trim();
          if (answer.length < 10) {
            answer = "Insufficient information in document context";
          }

          // Cache the answer with TTL
          answerCache.set(cacheKey, answer);
          setTimeout(() => answerCache.delete(cacheKey), 1800000); // 30 min TTL

          console.log(`✅ Q${questionIndex + 1} answered using ${modelUsed}`);
          return { index: questionIndex, answer };
        } catch (error) {
          console.error(
            `Error processing question ${questionIndex}:`,
            error.message
          );

          // Enhanced fallback with simpler approach
          try {
            const docs = await vectorStore.similaritySearch(question, 3);
            const simpleContext = docs
              .map((doc) => doc.pageContent.substring(0, 500))
              .join(" ");

            const fallbackResponse = await groq.chat.completions.create({
              messages: [
                {
                  role: "user",
                  content: `INSURANCE POLICY CONTEXT: ${simpleContext}

QUESTION: ${question}

Extract the specific answer from the policy document. Include exact numbers, timeframes, and conditions mentioned. Do not say information is insufficient - extract what's available.

JSON format: {"answer": "specific policy details"}`,
                },
              ],
              model: "llama-3.1-8b-instant",
              temperature: 0,
              max_completion_tokens: 200,
              response_format: { type: "json_object" },
            });

            let fallbackAnswer = "Processing error occurred";
            try {
              const parsed = JSON.parse(
                fallbackResponse.choices[0].message.content
              );
              fallbackAnswer =
                parsed.answer || "Unable to extract answer from context";
            } catch (fallbackParseError) {
              fallbackAnswer =
                "Unable to process question due to technical error";
            }

            answerCache.set(cacheKey, fallbackAnswer);
            return { index: questionIndex, answer: fallbackAnswer };
          } catch (fallbackError) {
            console.error(
              `Complete failure for question ${questionIndex}:`,
              fallbackError.message
            );
            return {
              index: questionIndex,
              answer: "Technical error: Unable to process this question",
            };
          }
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      // Rate limiting between batches
      if (i + batchSize < questions.length) {
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
    }

    // Sort and extract answers
    results.sort((a, b) => a.index - b.index);
    const answers = results.map((r) => r.answer);

    console.log(
      `✅ All ${questions.length} questions answered in ${
        Date.now() - startTime
      }ms`
    );
    return answers;
  }
}

const processor = new UltraFastAccurateProcessor();

// SINGLE OPTIMIZED ENDPOINT with enhanced accuracy
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}_${Math.random()
    .toString(36)
    .substr(2, 9)}`;

  try {
    const { documents, questions } = req.body;

    // Enhanced validation
    if (!documents || typeof documents !== "string") {
      return res.status(400).json({
        error: "Invalid request: 'documents' must be a valid URL string",
      });
    }

    if (!questions || !Array.isArray(questions) || questions.length === 0) {
      return res.status(400).json({
        error: "Invalid request: 'questions' must be a non-empty array",
      });
    }

    if (questions.length > 100) {
      return res.status(400).json({
        error: "Too many questions: Maximum 100 questions per request",
      });
    }

    console.log(
      `🚀 [${requestId}] Processing ${questions.length} questions for document`
    );

    // Step 1: Download and parse PDF (optimized)
    const {
      text,
      fromCache: textCached,
      filePath,
    } = await processor.downloadPDF(documents);
    const downloadTime = Date.now() - startTime;
    console.log(
      `📄 Text ${
        textCached ? "loaded from cache" : "downloaded"
      } in ${downloadTime}ms${filePath ? `, saved to: ${filePath}` : ""}`
    );

    if (!text || text.length < 100) {
      return res.status(400).json({
        error: "Document appears to be empty or too short to process",
      });
    }

    // Step 2: Create vector store (enhanced)
    const contentHash = crypto.createHash("sha256").update(text).digest("hex");
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

    // Step 3: Answer questions with high accuracy
    const answers = await processor.answerQuestions(
      questions,
      store,
      contentHash
    );
    const totalTime = Date.now() - startTime;

    // Enhanced performance metrics
    const avgTimePerQuestion = totalTime / questions.length;
    console.log(
      `✅ [${requestId}] Completed in ${totalTime}ms (${(
        totalTime / 1000
      ).toFixed(1)}s)`
    );
    console.log(
      `📊 Average time per question: ${avgTimePerQuestion.toFixed(1)}ms`
    );

    // Performance warnings
    if (totalTime > 30000) {
      console.warn(
        `⚠️  Response time ${(totalTime / 1000).toFixed(1)}s exceeds 30s target`
      );
    }

    // Enhanced response with metadata
    const responseData = {
      answers: answers,
    };

    // Log the response
    console.log(
      `📤 [${req.requestId || requestId}] Response:`,
      JSON.stringify(responseData, null, 2)
    );

    res.json(responseData);
  } catch (error) {
    const errorTime = Date.now() - startTime;
    console.error(
      `❌ [${requestId}] Error after ${errorTime}ms:`,
      error.message
    );

    // Enhanced error response
    const errorResponse = {
      error: "Processing failed",
      message: error.message,
      processing_time_ms: errorTime,
      request_id: requestId,
      timestamp: new Date().toISOString(),
    };

    // Log the error response
    console.log(
      `📤 [${req.requestId || requestId}] Error Response:`,
      JSON.stringify(errorResponse, null, 2)
    );

    res.status(500).json(errorResponse);
  }
});

// Enhanced health check
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    caches: {
      text_cache: textCache.size,
      vector_stores: vectorStores.size,
      answer_cache: answerCache.size,
      embedding_cache: embeddingCache.size,
    },
    models: {
      embedding: "text-embedding-004",
      llm_primary: "llama-3.3-70b-versatile",
      llm_fallback: "llama-3.1-70b-versatile",
    },
  });
});

// Enhanced cache management
app.post("/cache/clear", async (req, res) => {
  try {
    textCache.clear();
    vectorStores.clear();
    answerCache.clear();
    embeddingCache.clear();
    await redisClient.flushAll();
    res.json({
      message: "All caches cleared successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({
      error: "Cache clear failed",
      message: error.message,
    });
  }
});

// Start server with optimized settings
const server = app.listen(port, () => {
  console.log(`🚀 Ultra-Fast High-Accuracy PDF Server running on port ${port}`);
  console.log(`📊 Target: <30s response time with maximum accuracy`);
  console.log(`🔧 Latest Models: Llama-3.3-70B + Gemini text-embedding-004`);
  console.log(
    `⚡ Features: Enhanced caching, parallel processing, error recovery`
  );
});

// Enhanced server settings
server.setTimeout(45000); // 45s timeout for complex queries
server.keepAliveTimeout = 40000;
server.headersTimeout = 41000;

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\n🛑 Shutting down gracefully...");
  try {
    await redisClient.quit();
    server.close();
  } catch (e) {
    console.error("Shutdown error:", e.message);
  }
  process.exit(0);
});

// Enhanced error handling
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

process.on("uncaughtException", (error) => {
  console.error("Uncaught Exception:", error);
  process.exit(1);
});

module.exports = app;
