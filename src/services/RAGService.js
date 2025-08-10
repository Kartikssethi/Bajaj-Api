const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

class RAGService {
  constructor() {
    this.embeddings = null;
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1200, // Slightly larger chunks for speed
      chunkOverlap: 300,
      separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    });
    
    this.vectorStoreCache = new Map(); // Immediate memory cache
    this.embeddingCache = new Map();
    this.redisClient = null;
    
    // Initialize embeddings and Redis
    this.initializeServices();
    
    // Ensure FAISS stores directory exists
    this.faissStoresDir = path.join(process.cwd(), "faiss-stores");
    if (!fs.existsSync(this.faissStoresDir)) {
      fs.mkdirSync(this.faissStoresDir, { recursive: true });
    }
  }

  async initializeServices() {
    try {
      // Initialize Google embeddings
      if (process.env.GOOGLE_API_KEY) {
        this.embeddings = new GoogleGenerativeAIEmbeddings({
          apiKey: process.env.GOOGLE_API_KEY,
          model: "models/text-embedding-004",
        });
        console.log("✅ Google embeddings initialized");
      } else {
        console.warn("⚠️  GOOGLE_API_KEY not found - RAG functionality will be limited");
      }

      // Initialize Redis
      try {
        this.redisClient = require("../config/database");
        await this.redisClient.ping();
        console.log("✅ Redis connected for RAG caching");
      } catch (error) {
        console.warn("⚠️  Redis not available for RAG caching:", error.message);
      }
    } catch (error) {
      console.error("❌ Error initializing RAG services:", error.message);
    }
  }

  async createDocumentVectorStore(text, documentHash, requestId) {
    // Check memory cache first for immediate speed
    if (this.vectorStoreCache.has(documentHash)) {
      console.log(`[${requestId}] Vector store loaded from memory cache (instant)`);
      return { vectorStore: this.vectorStoreCache.get(documentHash), fromCache: true };
    }

    if (!this.embeddings) {
      throw new Error("Embeddings service not available - please set GOOGLE_API_KEY");
    }

    console.log(`[${requestId}] Creating vector store for ${(text.length / 1000).toFixed(1)}k chars...`);
    const startTime = Date.now();

    // Fast preprocessing
    const preprocessedText = this.preprocessText(text);

    // Create documents with enhanced metadata (like temp.js)
    const docs = await this.textSplitter.createDocuments([preprocessedText]);
    
    docs.forEach((doc, index) => {
      doc.metadata = {
        chunk_id: index,
        char_start: index * 1300,
        content_preview: doc.pageContent.substring(0, 100) + "...",
        documentHash
      };
    });

    console.log(`[${requestId}] Created ${docs.length} enhanced chunks in ${Date.now() - startTime}ms`);

    // Fast embedding and vector store creation
    const embeddingStart = Date.now();
    const vectorStore = await FaissStore.fromDocuments(docs, this.embeddings);
    console.log(`[${requestId}] Vector store created in ${Date.now() - embeddingStart}ms`);

    // Immediate memory cache with TTL (like temp.js)
    this.vectorStoreCache.set(documentHash, vectorStore);
    setTimeout(() => this.vectorStoreCache.delete(documentHash), 3600000); // 1 hour TTL

    // Optional background save to disk (non-blocking)
    this.saveVectorStoreBackground(vectorStore, documentHash, requestId);

    const totalTime = Date.now() - startTime;
    console.log(`[${requestId}] Total vector store creation: ${totalTime}ms`);

    return { vectorStore, fromCache: false };
  }

  async saveVectorStoreBackground(vectorStore, documentHash, requestId) {
    try {
      const vectorStorePath = path.join(this.faissStoresDir, documentHash);
      await vectorStore.save(vectorStorePath);
      
      if (this.redisClient) {
        await this.redisClient.setEx(`vectorstore:${documentHash}`, 7200, "stored");
      }
      console.log(`[${requestId}] Vector store saved to disk (background)`);
    } catch (error) {
      console.warn(`[${requestId}] Background save failed: ${error.message}`);
    }
  }

  async searchSimilarChunks(vectorStore, query, k = 5, requestId) {
    if (!vectorStore) {
      throw new Error("Vector store not available");
    }

    console.log(`[${requestId}] Searching for similar chunks (k=${k}): "${query.substring(0, 50)}..."`);
    
    try {
      const results = await vectorStore.similaritySearch(query, k);
      console.log(`[${requestId}] Found ${results.length} similar chunks`);
      
      return results.map((result, index) => ({
        content: result.pageContent,
        metadata: result.metadata,
        similarity: result.score || 0,
        rank: index + 1
      }));
    } catch (error) {
      console.error(`[${requestId}] Error searching similar chunks: ${error.message}`);
      throw new Error(`Similarity search failed: ${error.message}`);
    }
  }

  async getEnhancedContext(vectorStore, question, maxChunks = 8, requestId) {
    if (!vectorStore) {
      throw new Error("Vector store not available for context retrieval");
    }

    try {
      // Fast similarity search (like temp.js approach)
      let docs = await vectorStore.similaritySearch(question, maxChunks);

      // If not enough results, try keyword variations (from temp.js)
      if (docs.length < 3) {
        const keywordVariations = this.generateKeywordVariations(question);
        for (const variation of keywordVariations) {
          const additionalDocs = await vectorStore.similaritySearch(variation, 5);
          docs = [...docs, ...additionalDocs];
          if (docs.length >= 5) break;
        }
      }

      // Remove duplicates (from temp.js)
      const uniqueDocs = docs.filter(
        (doc, index, self) =>
          index === self.findIndex((d) => d.pageContent === doc.pageContent)
      );

      if (uniqueDocs.length === 0) {
        console.warn(`[${requestId}] No relevant chunks found for question`);
        return "";
      }

      // Fast context assembly
      const context = uniqueDocs.map((doc) => doc.pageContent).join("\n\n");

      console.log(`[${requestId}] Enhanced context: ${uniqueDocs.length} chunks (${context.length} chars)`);
      
      return context;
    } catch (error) {
      console.error(`[${requestId}] Error creating enhanced context: ${error.message}`);
      throw new Error(`Context creation failed: ${error.message}`);
    }
  }

  generateKeywordVariations(question) {
    const variations = [];
    const lowerQuestion = question.toLowerCase();

    const keyWords = lowerQuestion
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 2)
      .slice(0, 5);

    if (keyWords.length > 1) {
      variations.push(keyWords.join(" "));
      variations.push(keyWords.slice(0, 3).join(" "));
    }

    variations.push(question);

    return variations.slice(0, 2);
  }

  preprocessText(text) {
    return text
      .replace(
        /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu,
        ""
      )
      .replace(/\s+/g, " ")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  async embedText(text, requestId) {
    if (!this.embeddings) {
      throw new Error("Embeddings service not available");
    }

    const textHash = crypto.createHash("sha256").update(text).digest("hex");
    
    // Check memory cache
    if (this.embeddingCache.has(textHash)) {
      console.log(`[${requestId}] Embedding loaded from memory cache`);
      return this.embeddingCache.get(textHash);
    }

    // Check Redis cache
    if (this.redisClient) {
      try {
        const cachedEmbedding = await this.redisClient.get(`embedding:${textHash}`);
        if (cachedEmbedding) {
          const embedding = JSON.parse(cachedEmbedding);
          this.embeddingCache.set(textHash, embedding);
          console.log(`[${requestId}] Embedding loaded from Redis cache`);
          return embedding;
        }
      } catch (error) {
        console.warn(`[${requestId}] Redis embedding cache miss: ${error.message}`);
      }
    }

    // Create new embedding
    console.log(`[${requestId}] Creating new embedding for text (${text.length} chars)`);
    const embedding = await this.embeddings.embedQuery(text);
    
    // Cache the embedding
    this.embeddingCache.set(textHash, embedding);
    
    if (this.redisClient) {
      try {
        await this.redisClient.setEx(`embedding:${textHash}`, 7200, JSON.stringify(embedding));
        console.log(`[${requestId}] Embedding cached in Redis`);
      } catch (error) {
        console.warn(`[${requestId}] Failed to cache embedding in Redis: ${error.message}`);
      }
    }

    return embedding;
  }

  async clearCaches() {
    // Clear memory caches
    this.vectorStoreCache.clear();
    this.embeddingCache.clear();
    
    // Clear Redis caches
    if (this.redisClient) {
      try {
        const keys = await this.redisClient.keys("vectorstore:*");
        const embeddingKeys = await this.redisClient.keys("embedding:*");
        const allKeys = [...keys, ...embeddingKeys];
        
        if (allKeys.length > 0) {
          await this.redisClient.del(allKeys);
          console.log(`Cleared ${allKeys.length} RAG cache entries from Redis`);
        }
      } catch (error) {
        console.warn("Failed to clear RAG caches from Redis:", error.message);
      }
    }
    
    console.log("RAG caches cleared");
  }

  async getStorageStats() {
    const stats = {
      memoryCache: {
        vectorStores: this.vectorStoreCache.size,
        embeddings: this.embeddingCache.size
      },
      diskStorage: {
        faissStores: 0,
        totalSizeMB: 0
      },
      redisCache: {
        vectorStores: 0,
        embeddings: 0
      }
    };

    // Check disk storage
    try {
      if (fs.existsSync(this.faissStoresDir)) {
        const files = fs.readdirSync(this.faissStoresDir);
        stats.diskStorage.faissStores = files.filter(f => f.endsWith('.faiss')).length;
        
        let totalSize = 0;
        for (const file of files) {
          const filePath = path.join(this.faissStoresDir, file);
          const stat = fs.statSync(filePath);
          totalSize += stat.size;
        }
        stats.diskStorage.totalSizeMB = (totalSize / 1024 / 1024).toFixed(2);
      }
    } catch (error) {
      console.warn("Error reading FAISS storage stats:", error.message);
    }

    // Check Redis cache
    if (this.redisClient) {
      try {
        const vectorStoreKeys = await this.redisClient.keys("vectorstore:*");
        const embeddingKeys = await this.redisClient.keys("embedding:*");
        stats.redisCache.vectorStores = vectorStoreKeys.length;
        stats.redisCache.embeddings = embeddingKeys.length;
      } catch (error) {
        console.warn("Error reading Redis cache stats:", error.message);
      }
    }

    return stats;
  }

  async cleanupOldStores(maxAge = 7 * 24 * 60 * 60 * 1000) { // 7 days default
    try {
      if (!fs.existsSync(this.faissStoresDir)) return;

      const files = fs.readdirSync(this.faissStoresDir);
      let cleanedCount = 0;

      for (const file of files) {
        const filePath = path.join(this.faissStoresDir, file);
        const stat = fs.statSync(filePath);
        const fileAge = Date.now() - stat.mtime.getTime();

        if (fileAge > maxAge) {
          fs.unlinkSync(filePath);
          cleanedCount++;
        }
      }

      if (cleanedCount > 0) {
        console.log(`Cleaned up ${cleanedCount} old FAISS store files`);
      }
    } catch (error) {
      console.warn("Error during FAISS store cleanup:", error.message);
    }
  }
}

module.exports = RAGService;
