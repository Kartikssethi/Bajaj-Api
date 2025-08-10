const express = require('express');
const router = express.Router();
const cache = require('../utils/cache');
const RAGService = require('../services/RAGService');

// Get cache statistics including RAG
router.get('/stats', async (req, res) => {
  try {
    const stats = cache.getStats();
    
    // Get RAG storage stats
    const ragService = new RAGService();
    const ragStats = await ragService.getStorageStats();
    
    res.json({
      success: true,
      data: {
        ...stats,
        rag: ragStats
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Clear all cache including RAG
router.delete('/clear', async (req, res) => {
  try {
    const clearedCount = cache.clearAll();
    
    // Clear RAG caches
    const ragService = new RAGService();
    await ragService.clearCaches();
    
    res.json({
      success: true,
      message: `Cleared ${clearedCount} cache entries and RAG storage`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get cache keys
router.get('/keys', (req, res) => {
  try {
    const keys = cache.getKeys();
    res.json({
      success: true,
      data: keys,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get specific cache entry
router.get('/:key', (req, res) => {
  try {
    const { key } = req.params;
    const value = cache.get(key);
    
    if (value === undefined) {
      return res.status(404).json({
        success: false,
        error: 'Cache key not found',
        timestamp: new Date().toISOString()
      });
    }
    
    res.json({
      success: true,
      data: {
        key,
        value,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Delete specific cache entry
router.delete('/:key', (req, res) => {
  try {
    const { key } = req.params;
    const deleted = cache.delete(key);
    
    if (deleted) {
      res.json({
        success: true,
        message: `Cache key '${key}' deleted successfully`,
        timestamp: new Date().toISOString()
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Cache key not found',
        timestamp: new Date().toISOString()
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// RAG-specific endpoints
router.get('/rag/stats', async (req, res) => {
  try {
    const ragService = new RAGService();
    const stats = await ragService.getStorageStats();
    
    res.json({
      success: true,
      data: stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

router.delete('/rag/clear', async (req, res) => {
  try {
    const ragService = new RAGService();
    await ragService.clearCaches();
    
    res.json({
      success: true,
      message: 'RAG caches and vector stores cleared successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

router.delete('/rag/cleanup', async (req, res) => {
  try {
    const maxAge = req.query.maxAge ? parseInt(req.query.maxAge) : 7 * 24 * 60 * 60 * 1000; // 7 days default
    const ragService = new RAGService();
    await ragService.cleanupOldStores(maxAge);
    
    res.json({
      success: true,
      message: `Cleaned up old FAISS stores older than ${Math.round(maxAge / (24 * 60 * 60 * 1000))} days`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;
