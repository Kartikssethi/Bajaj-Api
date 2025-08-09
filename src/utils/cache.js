// Try to connect to Redis, but make it optional
let redisClient = null;
try {
  redisClient = require("../config/database");
} catch (error) {
  console.warn(
    "Redis not available, using in-memory cache only:",
    error.message
  );
}

// In-memory caches
const textCache = new Map();
const bufferCache = new Map();
const answerCache = new Map();
const embeddingCache = new Map();

// Counters
let requestCounter = 0;
let imageProcessingCounter = 0;

const clearAllCaches = async () => {
  textCache.clear();
  bufferCache.clear();
  answerCache.clear();
  embeddingCache.clear();
  requestCounter = 0;
  imageProcessingCounter = 0;

  if (redisClient) {
    try {
      await redisClient.flushAll();
    } catch (error) {
      console.warn("Redis cache clear failed:", error.message);
    }
  }
};

// Cache management functions
const get = (key) => {
  if (textCache.has(key)) return textCache.get(key);
  if (bufferCache.has(key)) return bufferCache.get(key);
  if (answerCache.has(key)) return answerCache.get(key);
  if (embeddingCache.has(key)) return embeddingCache.get(key);
  return undefined;
};

const deleteKey = (key) => {
  let deleted = false;
  if (textCache.delete(key)) deleted = true;
  if (bufferCache.delete(key)) deleted = true;
  if (answerCache.delete(key)) deleted = true;
  if (embeddingCache.delete(key)) deleted = true;
  return deleted;
};

const getKeys = () => {
  const keys = new Set();
  textCache.forEach((_, key) => keys.add(key));
  bufferCache.forEach((_, key) => keys.add(key));
  answerCache.forEach((_, key) => keys.add(key));
  embeddingCache.forEach((_, key) => keys.add(key));
  return Array.from(keys);
};

const getStats = () => {
  return {
    totalEntries:
      textCache.size +
      bufferCache.size +
      answerCache.size +
      embeddingCache.size,
    textCache: textCache.size,
    bufferCache: bufferCache.size,
    answerCache: answerCache.size,
    embeddingCache: embeddingCache.size,
    requestCounter,
    imageProcessingCounter,
  };
};

const clearAll = () => {
  const totalEntries =
    textCache.size + bufferCache.size + answerCache.size + embeddingCache.size;
  clearAllCaches();
  return totalEntries;
};

module.exports = {
  textCache,
  bufferCache,
  answerCache,
  embeddingCache,
  requestCounter,
  imageProcessingCounter,
  clearAllCaches,
  incrementRequestCounter: () => ++requestCounter,
  incrementImageCounter: () => ++imageProcessingCounter,
  get,
  delete: deleteKey,
  getKeys,
  getStats,
  clearAll,
};
