const express = require("express");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { Groq } = require("groq-sdk");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const redis = require("redis");
const crypto = require("crypto");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "100mb" }));


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

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Initialize Google Generative AI client for Gemini 2.5 Flash
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// Use Google Embedding API with GOOGLE_EMBEDDING_KEY for Gemini embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_EMBEDDING_KEY,
  model: "text-embedding-004", // Latest stable model
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

const vectorStores = new Map();
const textCache = new Map();
const answerCache = new Map();
const embeddingCache = new Map();

// Ultra-optimized processor with latest models
class LLMgobrr {
  constructor() {
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1200, // smaller chunks for better precision
      chunkOverlap: 300, // higher overlap to ensure important info isn't split
      separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    });

    // connection pool for parallel processing
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
    const variations = [];
    const lowerQuestion = question.toLowerCase();

    // extract key words (3+ characters) and create simple variations
    const keyWords = lowerQuestion
      .replace(/[^\w\s]/g, " ") // remove punctuation
      .split(/\s+/)
      .filter((word) => word.length > 2) //  meaningful words
      .slice(0, 5); // limit to top 5 keywords

    // create variations by combining keywords
    if (keyWords.length > 1) {
      variations.push(keyWords.join(" ")); 
      variations.push(keyWords.slice(0, 3).join(" ")); 
    }

    variations.push(question);

    return variations.slice(0, 2); 
  }

  async downloadPDF(url) {
    const urlHash = crypto.createHash("sha256").update(url).digest("hex");
    const pdfFileName = `pdf_${urlHash}.pdf`;
    const pdfFilePath = path.join(__dirname, "temp", pdfFileName);

    if (!fs.existsSync(path.join(__dirname, "temp"))) {
      fs.mkdirSync(path.join(__dirname, "temp"), { recursive: true });
    }

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
      `Downloaded in ${Date.now() - startTime}ms, size: ${(
        buffer.length /
        1024 /
        1024
      ).toFixed(2)}MB`
    );

    try {
      fs.writeFileSync(pdfFilePath, buffer);
      console.log(`PDF saved to: ${pdfFilePath}`);
    } catch (saveError) {
      console.warn(
        `Failed to save PDF to temp folder: ${saveError.message}`
      );
    }

    const parseStart = Date.now();
    const data = await pdfParse(buffer, {
      max: 0,
      version: "v1.10.100",
      normalizeWhitespace: true,
      disableCombineTextItems: false,
    });

    console.log(
      `Parsed PDF in ${Date.now() - parseStart}ms, ${
        data.numpages
      } pages, ${(data.text.length / 1000).toFixed(1)}k chars`
    );

    textCache.set(urlHash, data.text);
    try {
      const compressed = require("zlib").gzipSync(data.text).toString("base64");
      await redisClient.setEx(`pdf:${urlHash}`, 14400, compressed); 
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
    // Universal text preprocessing for any document type
    return (
      text
        // Normalize whitespace but preserve structure
        .replace(/\s+/g, " ")
        // Ensure section headers are properly spaced
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        // Clean up common formatting issues
        .replace(/\n{3,}/g, "\n\n") // Max 2 consecutive newlines
        .trim()
    );
  }

  async answerQuestions(questions, vectorStore, contentHash) {
    console.log(
      `Processing ${questions.length} questions`
    );
    const startTime = Date.now();

    const allQuestions = questions.map((question, index) => ({
      question,
      originalIndex: index,
    }));

    const results = await this.processQuestionsWithGeminiBatch(
      allQuestions,
      vectorStore,
      contentHash
    );
    const answers = results.map((r) => r.answer);

    console.log(
      `All ${questions.length} questions answered in ${
        Date.now() - startTime
      }ms`
    );
    return answers;
  }

  async processQuestionsWithGeminiBatch(questions, vectorStore, contentHash) {
    console.log(
      `Processing ${questions.length} questions`
    );

    const batchPromises = questions.map(async ({ question, originalIndex }) => {
      const cacheKey = `answer:${contentHash}:${question}`;

      if (answerCache.has(cacheKey)) {
        console.log(`Cache hit for question ${originalIndex + 1}`);
        return { originalIndex, answer: answerCache.get(cacheKey) };
      }

      try {
        const context = await this.getEnhancedContext(question, vectorStore);
        const prompt = this.buildPrompt(question, context);

        let answer;
        try {
          const model = genAI.getGenerativeModel({
            model: "gemini-2.5-flash",
            generationConfig: {
              temperature: 0.05,
              maxOutputTokens: 400,
              topP: 0.85,
              responseMimeType: "application/json",
            },
          });

          const result = await model.generateContent(prompt);
          const responseText = result.response.text();
          answer = this.parseResponse(responseText);

          console.log(`✅ Gemini answered question ${originalIndex + 1}`);
        } catch (geminiError) {
          console.warn(
            `⚠️ Gemini failed for question ${
              originalIndex + 1
            }, falling back to Groq...`
          );

          // Fallback to Groq
          const groqModels = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
          ];

          let response;
          for (const model of groqModels) {
            try {
              response = await groq.chat.completions.create({
                messages: [{ role: "user", content: prompt }],
                model: model,
                temperature: 0.05,
                max_completion_tokens: 400,
                response_format: { type: "json_object" },
                top_p: 0.85,
                frequency_penalty: 0.1,
                presence_penalty: 0.1,
              });
              break;
            } catch (modelError) {
              console.warn(`Groq model ${model} failed, trying next...`);
              continue;
            }
          }

          if (response) {
            answer = this.parseResponse(response.choices[0].message.content);
            console.log(
              `✅ Groq fallback answered question ${originalIndex + 1}`
            );
          } else {
            throw new Error("All models failed");
          }
        }

        // Cache the answer
        answerCache.set(cacheKey, answer);
        return { originalIndex, answer };
      } catch (error) {
        console.error(
          `❌ Error processing question ${originalIndex + 1}:`,
          error
        );
        return {
          originalIndex,
          answer: "Sorry, I encountered an error processing this question.",
        };
      }
    });

    // Wait for all questions to be processed in parallel
    const results = await Promise.all(batchPromises);
    return results;
  }

  async processQuestionsWithGroqBatch(questions, vectorStore, contentHash) {
    console.log(
      `🔥 Processing ${questions.length} questions with Groq parallel batch processing...`
    );

    const groqModels = [
      "llama-3.3-70b-versatile",
      "llama-3.1-70b-versatile",
      "llama-3.1-8b-instant",
    ];

    // Process all questions in parallel for maximum speed
    const batchPromises = questions.map(async ({ question, originalIndex }) => {
      const cacheKey = `answer:${contentHash}:${question}`;

      // Check cache first
      if (answerCache.has(cacheKey)) {
        console.log(`💾 Cache hit for question ${originalIndex + 1}`);
        return { originalIndex, answer: answerCache.get(cacheKey) };
      }

      try {
        const context = await this.getEnhancedContext(question, vectorStore);
        const prompt = this.buildPrompt(question, context);

        let response;
        for (const model of groqModels) {
          try {
            response = await groq.chat.completions.create({
              messages: [{ role: "user", content: prompt }],
              model: model,
              temperature: 0.05,
              max_completion_tokens: 400,
              response_format: { type: "json_object" },
              top_p: 0.85,
              frequency_penalty: 0.1,
              presence_penalty: 0.1,
            });
            break;
          } catch (error) {
            console.log(
              `⚠️ Groq model ${model} failed for Q${
                originalIndex + 1
              }, trying next...`
            );
          }
        }

        if (!response) {
          throw new Error("All Groq models failed");
        }

        const answer = JSON.parse(response.choices[0].message.content).answer;
        answerCache.set(cacheKey, answer);

        console.log(`✅ Groq answered question ${originalIndex + 1}`);
        return { originalIndex, answer };
      } catch (error) {
        console.error(
          `❌ Error processing question ${originalIndex + 1}:`,
          error
        );
        return {
          originalIndex,
          answer: "Sorry, I encountered an error processing this question.",
        };
      }
    });

    // Wait for all questions to be processed in parallel
    const results = await Promise.all(batchPromises);
    return results;
  }

  async processQuestionsWithGroq(questionObjects, vectorStore, contentHash) {
    if (questionObjects.length === 0) return [];

    console.log(`🔥 Groq processing ${questionObjects.length} questions...`);
    const results = [];

    // Groq models optimized for speed and accuracy
    const groqModels = [
      "llama-3.1-8b-instant",
      "llama-3.1-70b-versatile",
      "llama-3.1-8b-instant",
    ];

    for (const { question, originalIndex } of questionObjects) {
      const cacheKey = `groq:${contentHash}:${crypto
        .createHash("sha256")
        .update(question)
        .digest("hex")}`;

      if (answerCache.has(cacheKey)) {
        results.push({ originalIndex, answer: answerCache.get(cacheKey) });
        continue;
      }

      try {
        const context = await this.getEnhancedContext(question, vectorStore);
        const prompt = this.buildPrompt(question, context);

        let response;
        for (const model of groqModels) {
          try {
            response = await groq.chat.completions.create({
              messages: [{ role: "user", content: prompt }],
              model: model,
              temperature: 0.05,
              max_completion_tokens: 400,
              response_format: { type: "json_object" },
              top_p: 0.85,
              frequency_penalty: 0.1,
              presence_penalty: 0.1,
            });
            break;
          } catch (modelError) {
            console.warn(`Groq model ${model} failed, trying next...`);
            continue;
          }
        }

        const answer = this.parseResponse(
          response?.choices[0]?.message?.content
        );
        answerCache.set(cacheKey, answer);
        results.push({ originalIndex, answer });
      } catch (error) {
        console.error(
          `Groq error for question ${originalIndex}:`,
          error.message
        );
        results.push({
          originalIndex,
          answer: "Groq processing error occurred",
        });
      }
    }

    console.log(`✅ Groq completed ${results.length} questions`);
    return results;
  }

  async getEnhancedContext(question, vectorStore) {
    // Multi-stage search for comprehensive coverage
    let docs = await vectorStore.similaritySearch(question, 8);

    // If first search doesn't yield enough, try keyword variations
    if (docs.length < 3) {
      const keywordVariations = this.generateKeywordVariations(question);
      for (const variation of keywordVariations) {
        const additionalDocs = await vectorStore.similaritySearch(variation, 5);
        docs = [...docs, ...additionalDocs];
        if (docs.length >= 5) break;
      }
    }

    // Remove duplicates and get comprehensive context
    const uniqueDocs = docs.filter(
      (doc, index, self) =>
        index === self.findIndex((d) => d.pageContent === doc.pageContent)
    );

    return uniqueDocs.map((doc) => doc.pageContent).join("\n\n");
  }

  buildPrompt(question, context) {
    return `You are an expert document analyst. Extract the precise answer from the document context provided.

  DOCUMENT CONTEXT:
  ${context.substring(0, 3000)}

  QUESTION: ${question}

  CRITICAL INSTRUCTIONS:
  1. Extract the EXACT information from the document - do not say "context doesn't contain enough information" unless truly absent
  2. Look for specific numbers, dates, percentages, conditions, and definitions
  3. If you find partial information, provide what's available with specific details
  4. Include specific terms, timeframes, and conditions mentioned in the document
  5. For definitions, provide the complete definition as stated
  6. Be comprehensive but concise - include all relevant details from the document
  7. Write answers in clear, human-readable format with proper punctuation and grammar
  8. Focus on factual information directly from the document

  FORMAT: Respond with JSON: {"answer": "detailed answer with specific information from document"}

  ANSWER:`;
  }

  parseResponse(responseContent) {
    if (!responseContent) return "No response received";

    try {
      const parsed = JSON.parse(responseContent);
      return parsed.answer || "No answer found in context";
    } catch (parseError) {
      // Enhanced fallback parsing
      const answerMatch = responseContent.match(/"answer"\s*:\s*"([^"]+)"/);
      if (answerMatch) {
        return answerMatch[1];
      } else {
        // Extract meaningful content
        return (
          responseContent.replace(/[{}]/g, "").split(":").pop()?.trim() ||
          "Parse error occurred"
        );
      }
    }
  }
}

const processor = new LLMgobrr();

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
    providers: {
      embedding: "Google text-embedding-004 (GOOGLE_EMBEDDING_KEY)",
      llm_providers: {
        primary: "Google Gemini 2.5 Flash (GOOGLE_API_KEY)",
        fallback: "Groq llama-3.3-70b-versatile",
      },
      architecture: "Gemini Primary + Groq Fallback 🚀",
    },
    features: [
      "Gemini 2.5 Flash primary processing",
      "Groq fallback for reliability",
      "Enhanced caching",
      "Error recovery",
      "Parallel processing",
    ],
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
  console.log(`🚀 GEMINI + GROQ AI Server running on port ${port}`);
  console.log(`⚡ PRIMARY: Google Gemini 2.5 Flash (accuracy + speed)`);
  console.log(`🔄 FALLBACK: Groq llama-3.3-70b-versatile (reliability)`);
  console.log(`📊 Target: <5s for small batches, <15s for large batches`);
  console.log(
    `🔧 Embeddings: Google text-embedding-004 (GOOGLE_EMBEDDING_KEY)`
  );
  console.log(`🌟 Features: Parallel processing, smart fallback, caching`);
  console.log(`🎯 GEMINI POWERED: Maximum Accuracy + Speed!`);
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