const express = require('express');
const router = express.Router();
const cache = require('../utils/cache');

// Get cache statistics
router.get('/stats', (req, res) => {
  try {
    const stats = cache.getStats();
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

// Clear all cache
router.delete('/clear', (req, res) => {
  try {
    const clearedCount = cache.clearAll();
    res.json({
      success: true,
      message: `Cleared ${clearedCount} cache entries`,
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

module.exports = router;
