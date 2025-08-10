const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const compression = require("compression");
const morgan = require("morgan");
require("dotenv").config();

const routes = require("./src/routes");
const loggingMiddleware = require("./src/middleware/logging");

const app = express();
const port = process.env.PORT || 3000;

// Security middleware
app.use(helmet());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: {
    error: "Too many requests from this IP, please try again later.",
    timestamp: new Date().toISOString(),
  },
});
app.use(limiter);

// Compression middleware
app.use(compression());

// Logging middleware
app.use(morgan("combined"));
app.use(loggingMiddleware);

// CORS configuration
app.use(
  cors({
    origin:  "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"],
    credentials: true,
  })
);

// Body parsing middleware
app.use(express.json({ limit: "100mb" }));
app.use(express.urlencoded({ extended: true, limit: "100mb" }));

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: "1.0.0",
  });
});

// API routes
app.use(routes);

// Root endpoint
app.get("/", (req, res) => {
  res.json({
    message: "Bajaj API - Document Processing Service",
    version: "1.0.0",
    endpoints: {
      health: "/health",
      api: "",
      hackrx: "/hackrx/run (with RAG support)",
      cache: "/cache",
      ragStats: "/cache/rag/stats", 
      ragClear: "/cache/rag/clear",
      quiz: "/quiz/solve",
    },
    timestamp: new Date().toISOString(),
  });
});

// 404 handler - use a proper catch-all route
app.use((req, res) => {
  res.status(404).json({
    error: "Endpoint not found",
    message: `The requested endpoint ${req.originalUrl} does not exist`,
    availableEndpoints: {
      health: "/health",
      api: "",
      hackrx: "/hackrx/run (with RAG support)",
      cache: "/cache",
      ragStats: "/cache/rag/stats",
      ragClear: "/cache/rag/clear",
      quiz: "/quiz/solve",
    },
    timestamp: new Date().toISOString(),
  });
});

// Global error handler
app.use((error, req, res, next) => {
  console.error("Global error handler:", error);

  res.status(error.status || 500).json({
    error: "Internal server error",
    message:
      process.env.NODE_ENV === "production"
        ? "An unexpected error occurred"
        : error.message,
    timestamp: new Date().toISOString(),
    ...(process.env.NODE_ENV === "development" && { stack: error.stack }),
  });
});

// Start server
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
  console.log(`📊 Health check: http://localhost:${port}/health`);
  console.log(`🔗 API base: http://localhost:${port}`);
  console.log(`📝 Main endpoint (RAG): http://localhost:${port}/hackrx/run`);
  console.log(`🗄️  Cache management: http://localhost:${port}/cache`);
  console.log(`🧠 RAG stats: http://localhost:${port}/cache/rag/stats`);
});

module.exports = app;
