const express = require('express');
const router = express.Router();

const hackrxRoutes = require('./hackrx');
const healthRoutes = require('./health');
const cacheRoutes = require('./cache');
const quizRoutes = require('./quiz');

// Mount route modules
router.use('/hackrx', hackrxRoutes);
router.use('/health', healthRoutes);
router.use('/cache', cacheRoutes);
router.use('/quiz', quizRoutes);

module.exports = router;
