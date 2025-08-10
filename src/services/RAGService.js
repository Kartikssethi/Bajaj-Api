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
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    });
    
    this.vectorStoreCache = new Map();
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
    console.log(`[${requestId}] Creating vector store for document: ${documentHash.substring(0, 8)}...`);
    
    if (!this.embeddings) {
      throw new Error("Embeddings service not available - please set GOOGLE_API_KEY");
    }

    // Check if vector store already exists in cache
    if (this.vectorStoreCache.has(documentHash)) {
      console.log(`[${requestId}] Vector store loaded from memory cache`);
      return { vectorStore: this.vectorStoreCache.get(documentHash), fromCache: true };
    }

    // Check if vector store exists in Redis
    if (this.redisClient) {
      try {
        const cachedVectorStore = await this.redisClient.get(`vectorstore:${documentHash}`);
        if (cachedVectorStore) {
          const vectorStorePath = path.join(this.faissStoresDir, `${documentHash}.faiss`);
          if (fs.existsSync(vectorStorePath)) {
            console.log(`[${requestId}] Loading vector store from Redis cache`);
            const vectorStore = await FaissStore.load(vectorStorePath, this.embeddings);
            this.vectorStoreCache.set(documentHash, vectorStore);
            return { vectorStore, fromCache: true };
          }
        }
      } catch (error) {
        console.warn(`[${requestId}] Redis vector store cache miss: ${error.message}`);
      }
    }

    const startTime = Date.now();
    
    // Split text into chunks
    console.log(`[${requestId}] Splitting text into chunks...`);
    const chunks = await this.textSplitter.splitText(text);
    console.log(`[${requestId}] Created ${chunks.length} chunks`);

    // Create vector store from chunks
    console.log(`[${requestId}] Creating embeddings and vector store...`);
    const vectorStore = await FaissStore.fromTexts(
      chunks,
      chunks.map((_, i) => ({ chunkIndex: i, documentHash })),
      this.embeddings
    );

    // Save vector store to disk
    const vectorStorePath = path.join(this.faissStoresDir, documentHash);
    await vectorStore.save(vectorStorePath);
    console.log(`[${requestId}] Vector store saved to: ${vectorStorePath}`);

    // Cache in memory and Redis
    this.vectorStoreCache.set(documentHash, vectorStore);
    
    if (this.redisClient) {
      try {
        await this.redisClient.setEx(`vectorstore:${documentHash}`, 7200, "stored"); // Cache for 2 hours
        console.log(`[${requestId}] Vector store cached in Redis`);
      } catch (error) {
        console.warn(`[${requestId}] Failed to cache vector store in Redis: ${error.message}`);
      }
    }

    const creationTime = Date.now() - startTime;
    console.log(`[${requestId}] Vector store created in ${creationTime}ms`);

    return { vectorStore, fromCache: false };
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

  async getEnhancedContext(vectorStore, question, maxChunks = 5, requestId) {
    if (!vectorStore) {
      throw new Error("Vector store not available for context retrieval");
    }

    try {
      // Search for relevant chunks
      const similarChunks = await this.searchSimilarChunks(vectorStore, question, maxChunks, requestId);
      
      if (similarChunks.length === 0) {
        console.warn(`[${requestId}] No relevant chunks found for question`);
        return "";
      }

      // Combine chunks into context
      const context = similarChunks
        .map((chunk, index) => `[Chunk ${index + 1}]\n${chunk.content}`)
        .join("\n\n");

      console.log(`[${requestId}] Enhanced context created from ${similarChunks.length} chunks (${context.length} chars)`);
      
      return context;
    } catch (error) {
      console.error(`[${requestId}] Error creating enhanced context: ${error.message}`);
      throw new Error(`Context creation failed: ${error.message}`);
    }
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
