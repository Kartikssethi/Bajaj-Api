// package.json dependencies needed:
/*
{
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "multer": "^1.4.5-lts.1",
    "pdf-parse": "^1.1.1",
    "mammoth": "^1.6.0",
    "faiss-node": "^0.5.1",
    "@google/generative-ai": "^0.21.0",
    "ai": "^3.4.7",
    "@langchain/core": "^0.3.0",
    "@langchain/community": "^0.3.0",
    "@langchain/google-genai": "^0.1.0",
    "axios": "^1.7.7",
    "dotenv": "^16.4.5"
  }
}
*/

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

// Initialize Gemini with latest 2.5 Flash model
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// Initialize embeddings with latest model
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004", // Latest embedding model
});

// Store for FAISS instances (in production, use Redis or database)
const vectorStores = new Map();

class DocumentProcessor {
  constructor() {
    // Optimized for speed - smaller chunks, less overlap
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600, // Smaller chunks for faster processing
      chunkOverlap: 50, // Minimal overlap for speed
    });
  }

  async extractTextFromFile(filePath, fileType) {
    try {
      if (fileType === "application/pdf") {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdfParse(dataBuffer);
        return data.text;
      } else if (
        fileType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        const result = await mammoth.extractRawText({ path: filePath });
        return result.value;
      } else if (fileType === "text/plain") {
        return fs.readFileSync(filePath, "utf8");
      } else {
        throw new Error(`Unsupported file type: ${fileType}`);
      }
    } catch (error) {
      throw new Error(`Error extracting text: ${error.message}`);
    }
  }

  async downloadAndExtractFromUrl(url) {
    try {
      console.log("⚡ Downloading document...");
      const response = await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 15000, // 15 second timeout
        maxContentLength: 10 * 1024 * 1024, // 10MB limit for speed
      });
      const contentType = response.headers["content-type"];

      // Create temporary file
      const tempDir = path.join(__dirname, "temp");
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir);
      }

      const tempFilePath = path.join(tempDir, `temp_${Date.now()}.pdf`);
      fs.writeFileSync(tempFilePath, response.data);

      console.log("⚡ Extracting text...");
      const text = await this.extractTextFromFile(
        tempFilePath,
        "application/pdf"
      );

      // Clean up temp file immediately
      fs.unlinkSync(tempFilePath);

      return text;
    } catch (error) {
      throw new Error(
        `Error downloading and extracting from URL: ${error.message}`
      );
    }
  }

  async processAndStoreDocument(documentSource, storeId) {
    try {
      let text;

      console.log("⚡ Processing document...");
      // Check if documentSource is URL or file path
      if (documentSource.startsWith("http")) {
        text = await this.downloadAndExtractFromUrl(documentSource);
      } else {
        // Assume it's a file path
        text = await this.extractTextFromFile(
          documentSource,
          "application/pdf"
        );
      }

      console.log("⚡ Splitting text into chunks...");
      // Split text into chunks
      const docs = await this.textSplitter.createDocuments([text]);

      const chunks = docs
        .map((doc, index) => `--- Chunk ${index} ---\n${doc.pageContent}\n\n`)
        .join("");

      fs.writeFileSync("chunks.txt", chunks, "utf8");
      console.log("Chunks written to chunks.txt");

      console.log(`⚡ Creating embeddings for ${docs.length} chunks...`);
      // Create FAISS vector store with parallel processing
      const vectorStore = await FaissStore.fromDocuments(docs, embeddings);

      // Store the vector store
      vectorStores.set(storeId, vectorStore);

      console.log("✅ Document processed successfully!");
      return {
        success: true,
        chunksCount: docs.length,
        storeId: storeId,
      };
    } catch (error) {
      throw new Error(`Error processing document: ${error.message}`);
    }
  }

  async answerQuestions(questions, storeId) {
    try {
      const vectorStore = vectorStores.get(storeId);
      if (!vectorStore) {
        throw new Error(
          "Document not found. Please upload the document first."
        );
      }

      console.log(`⚡ Processing ${questions.length} questions...`);
      const answers = [];

      // Process questions in parallel for speed
      const answerPromises = questions.map(async (question, index) => {
        console.log(
          `⚡ Question ${index + 1}: ${question.substring(0, 50)}...`
        );

        // Retrieve only top 2 most relevant docs for speed
        const relevantDocs = await vectorStore.similaritySearch(question, 10);

        // Prepare context from relevant documents
        const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

        // Optimized prompt for faster response
        const prompt = `
You are an expert assistant. Use ONLY the following context to answer the question.

Context:
${context}

Question:
${question}

Instructions:
- Give a one-sentence answer that includes all critical details from the context.
- Limit to 35 words max.
- Do NOT add any explanation or repeat the question.
- If the answer is not in the context, try harder and infer from the document.You cannot say document not found unless you are 200% sure its not there and you cant infer from rest of document 
- Respond only with a JSON array: ["answer"]

Answer:
`;

        const result = await model.generateContent(prompt);
        let rawText = result.response.text().trim();

        // ✅ Clean up code block formatting (remove ```json and ```)
        rawText = rawText.replace(/```json|```/g, "").trim();

        let parsedAnswer;
        try {
          parsedAnswer = JSON.parse(rawText); // Try to parse cleaned string into an array
        } catch (e) {
          // If parsing fails, assume it's a plain string, wrap it in an array
          parsedAnswer = [rawText];
        }

        return { index, answer: parsedAnswer };
      });

      // Wait for all answers to complete
      const results = await Promise.all(answerPromises);

      // Sort by original order
      results.sort((a, b) => a.index - b.index);

      console.log("✅ All questions processed!");
      return results.flatMap((r) => r.answer); // flatten answers into a single array
    } catch (error) {
      throw new Error(`Error answering questions: ${error.message}`);
    }
  }
}

const documentProcessor = new DocumentProcessor();

// Routes

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "OK", timestamp: new Date().toISOString() });
});

// Upload and process document (file upload)
app.post("/upload", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const storeId = `doc_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    const result = await documentProcessor.processAndStoreDocument(
      req.file.path,
      storeId
    );

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      message: "Document processed successfully",
      storeId: storeId,
      chunksCount: result.chunksCount,
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Main endpoint matching the sample request format
app.post("/hackrx/run", async (req, res) => {
  const startTime = Date.now();

  try {
    // Validate authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res
        .status(401)
        .json({ error: "Missing or invalid authorization header" });
    }

    const { documents, questions } = req.body;

    if (!documents || !questions || !Array.isArray(questions)) {
      return res.status(400).json({
        error:
          "Invalid request format. Expected documents (string) and questions (array)",
      });
    }

    // Generate a unique store ID for this request
    const storeId = `hackrx_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    console.log(`🚀 HackRX Request started - ${questions.length} questions`);

    // Process the document
    console.log("📄 Processing document:", documents.substring(0, 100) + "...");
    await documentProcessor.processAndStoreDocument(documents, storeId);

    // Answer the questions
    console.log("❓ Answering questions...");
    const answers = await documentProcessor.answerQuestions(questions, storeId);

    // Clean up the vector store immediately for memory efficiency
    vectorStores.delete(storeId);

    const processingTime = Date.now() - startTime;
    console.log(`✅ Request completed in ${processingTime}ms`);

    // Return structured response matching the sample format
    res.json({
      answers: answers,
    });
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error(`❌ HackRX error after ${processingTime}ms:`, error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

// Answer questions for existing document
app.post("/answer/:storeId", async (req, res) => {
  try {
    const { storeId } = req.params;
    const { questions } = req.body;

    if (!questions || !Array.isArray(questions)) {
      return res
        .status(400)
        .json({ error: "Questions must be provided as an array" });
    }

    const answers = await documentProcessor.answerQuestions(questions, storeId);

    res.json({
      storeId: storeId,
      answers: answers,
    });
  } catch (error) {
    console.error("Answer error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Get available document stores
app.get("/stores", (req, res) => {
  const stores = Array.from(vectorStores.keys());
  res.json({ stores: stores, count: stores.length });
});

// Delete a document store
app.delete("/stores/:storeId", (req, res) => {
  const { storeId } = req.params;

  if (vectorStores.has(storeId)) {
    vectorStores.delete(storeId);
    res.json({ message: "Document store deleted successfully" });
  } else {
    res.status(404).json({ error: "Document store not found" });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error("Unhandled error:", error);
  res.status(500).json({
    error: "Internal server error",
    message: error.message,
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: "Endpoint not found" });
});

// Start server
app.listen(port, () => {
  console.log(`Document Q&A Server running on port ${port}`);
  console.log("Available endpoints:");
  console.log("  POST /hackrx/run - Main endpoint (matches sample format)");
  console.log("  POST /upload - Upload document file");
  console.log(
    "  POST /answer/:storeId - Answer questions for existing document"
  );
  console.log("  GET /stores - List document stores");
  console.log("  DELETE /stores/:storeId - Delete document store");
  console.log("  GET /health - Health check");
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\nShutting down gracefully...");
  process.exit(0);
});

module.exports = app;
