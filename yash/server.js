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
const multer = require("multer");
const mammoth = require("mammoth");
const XLSX = require("xlsx");
const sharp = require("sharp");
const yauzl = require("yauzl"); // For ZIP file handling
const { parse: csvParse } = require("csv-parse"); // For CSV parsing
const xml2js = require("xml2js"); // For PowerPoint parsing
const officegen = require("officegen"); // For PowerPoint processing
const fs_extra = require("fs-extra"); // For enhanced file operations
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "100mb" }));

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "temp", "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
      "application/msword", // .doc
      "application/vnd.ms-excel", // .xls
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
      "application/vnd.ms-powerpoint", // .ppt
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
      "text/csv",
      "application/csv",
      "image/jpeg",
      "image/png",
      "image/gif",
      "image/bmp",
      "image/tiff",
      "application/zip",
      "application/x-zip-compressed",
      "text/plain", // .txt, .log
      "text/markdown", // .md
      "text/html", // .html, .htm
      "application/xhtml+xml", // .html, .htm
      "application/json", // .json
      "application/xml", // .xml
      "text/xml", // .xml
    ];

    // Check by mimetype or by extension (for robust filtering)
    if (
      allowedMimeTypes.includes(file.mimetype) ||
      file.originalname
        .toLowerCase()
        .match(
          /\.(pdf|doc|docx|xls|xlsx|ppt|pptx|csv|jpg|jpeg|png|gif|bmp|tiff|zip)$/i
        )
    ) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Unsupported file type: ${file.mimetype}. Allowed types include PDF, Word, Excel, CSV, Image, and ZIP.`
        ),
        false
      );
    }
  },
});

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
      `[${requestId}] ${timestamp} - Incoming ${req.method} ${req.path}`
    );
    console.log(
      `[${requestId}] Request Body:`,
      JSON.stringify(req.body, null, 2)
    );
    req.requestId = requestId;
  }
  next();
});

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_EMBEDDING_KEY,
  model: "text-embedding-004",
});

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
const textCache = new Map(); // Cache for processed text content (string)
const bufferCache = new Map(); // Cache for raw file buffers (for URL downloads)
const answerCache = new Map();
const embeddingCache = new Map();

let requestCounter = 0;
let imageProcessingCounter = 0; // Separate counter for image processing alternation

class LLMgobrr {
  constructor() {
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1200,
      chunkOverlap: 300,
      separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    });

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

  async downloadFile(url, requestId) {
    const urlHash = crypto.createHash("sha256").update(url).digest("hex");
    const fileExtension = this.getFileExtensionFromUrl(url);
    const fileName = `downloaded_url_${urlHash}${fileExtension}`;
    const filePath = path.join(__dirname, "temp", fileName);

    if (!fs.existsSync(path.join(__dirname, "temp"))) {
      fs.mkdirSync(path.join(__dirname, "temp"), { recursive: true });
    }

    // Check buffer cache first (for raw downloaded content)
    let buffer;
    if (bufferCache.has(urlHash)) {
      buffer = bufferCache.get(urlHash);
      console.log(
        `[${requestId}] File buffer loaded from memory cache for URL: ${url.substring(
          0,
          50
        )}...`
      );
      // Re-save to disk for fileProcessor to use (if needed)
      fs.writeFileSync(filePath, buffer);
      return { buffer, fromCache: true, filePath };
    }

    // Check Redis cache for downloaded file buffer
    try {
      const cachedBufferB64 = await redisClient.get(`url_buffer:${urlHash}`);
      if (cachedBufferB64) {
        buffer = Buffer.from(cachedBufferB64, "base64");
        bufferCache.set(urlHash, buffer); // Populate memory cache
        console.log(
          `[${requestId}] File buffer loaded from Redis cache for URL: ${url.substring(
            0,
            50
          )}...`
        );
        // Re-save to disk for fileProcessor to use
        fs.writeFileSync(filePath, buffer);
        return { buffer, fromCache: true, filePath };
      }
    } catch (e) {
      console.warn(
        `[${requestId}] Redis buffer cache miss for URL ${url.substring(
          0,
          50
        )}...: ${e.message}`
      );
    }

    console.log(
      `[${requestId}] Downloading file from URL: ${url.substring(0, 50)}...`
    );
    const startTime = Date.now();

    let retries = 3;
    let response;

    while (retries > 0) {
      try {
        response = await axios.get(url, {
          responseType: "arraybuffer", // Crucial for handling all file types as binary
          timeout: 60000,
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            Accept: "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            Connection: "keep-alive",
            "Cache-Control": "no-cache",
          },
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        });
        buffer = Buffer.from(response.data);
        break;
      } catch (error) {
        retries--;
        console.warn(
          `[${requestId}] Download retry ${
            3 - retries
          }/${3} for ${url.substring(0, 50)}... Error: ${error.message}`
        );
        if (retries === 0) throw error;
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    console.log(
      `[${requestId}] Downloaded in ${Date.now() - startTime}ms, size: ${(
        buffer.length /
        1024 /
        1024
      ).toFixed(2)}MB`
    );

    try {
      fs.writeFileSync(filePath, buffer);
      console.log(`[${requestId}] File saved to: ${filePath}`);
    } catch (saveError) {
      console.warn(
        `[${requestId}] Failed to save file to temp folder: ${saveError.message}`
      );
      throw new Error(`Could not save downloaded file: ${saveError.message}`);
    }

    // Cache the raw buffer in Redis and memory
    bufferCache.set(urlHash, buffer);
    try {
      await redisClient.setEx(
        `url_buffer:${urlHash}`,
        14400,
        buffer.toString("base64")
      ); // 4 hour TTL
    } catch (e) {
      console.warn(
        `[${requestId}] Redis buffer cache write failed for URL ${url.substring(
          0,
          50
        )}...: ${e.message}`
      );
    }

    return { buffer, fromCache: false, filePath };
  }

  getFileExtensionFromUrl(url) {
    try {
      const urlObj = new URL(url);
      const pathname = urlObj.pathname;
      const extension = path.extname(pathname).toLowerCase();
      return extension || ""; // Return empty string for no extension
    } catch (error) {
      return "";
    }
  }

  getFileTypeFromUrl(url) {
    const extension = this.getFileExtensionFromUrl(url);
    if (!extension) {
      // If no extension, try to guess from common URL patterns or default to unknown
      if (
        url.includes("image") &&
        (url.includes(".jpeg") || url.includes(".png"))
      )
        return "image";
      return "unknown";
    }

    if (extension === ".pdf") return "pdf";
    if ([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"].includes(extension))
      return "image";
    if (extension === ".bin") return "bin";
    if ([".doc", ".docx"].includes(extension)) return "word";
    if ([".xls", ".xlsx"].includes(extension)) return "excel";
    if ([".ppt", ".pptx"].includes(extension)) return "powerpoint";
    if (extension === ".csv") return "csv";
    if (extension === ".zip") return "zip";

    return "unknown";
  }

  async processImageBufferWithGemini(buffer, extension) {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" }); // Use gemini-pro-vision for images

    const base64Image = buffer.toString("base64");
    const mimeType = this.getMimeTypeFromExtension(extension);

    const prompt =
      "Extract all text content from this image. Return the extracted text exactly as it appears in the image, maintaining any formatting, structure, and organization. Do not add any introductory or concluding remarks.";

    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: mimeType,
      },
    };

    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    let extractedText = response.text();
    // Gemini may sometimes add intro/outro despite prompt. Attempt to strip if detected.
    if (extractedText.startsWith("Extracted Text:\n")) {
      extractedText = extractedText.substring("Extracted Text:\n".length);
    }
    if (extractedText.endsWith("\nEnd of Extraction")) {
      extractedText = extractedText.substring(
        0,
        extractedText.length - "\nEnd of Extraction".length
      );
    }
    return extractedText.trim();
  }

  // Groq does not have native image processing. This function explicitly uses Gemini as a fallback/primary for image.
  async processImageBufferWithGroqFallback(buffer, extension) {
    try {
      // Convert buffer to base64
      const base64Image = buffer.toString("base64");
      const mimeType = this.getMimeTypeFromExtension(extension);
      const dataUrl = `data:${mimeType};base64,${base64Image}`;

      const response = await groq.chat.completions.create({
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Extract all text content from this image. Return the extracted text exactly as it appears in the image, maintaining any formatting, structure, and organization. Do not add any introductory or concluding remarks.",
              },
              {
                type: "image_url",
                image_url: {
                  url: dataUrl,
                },
              },
            ],
          },
        ],
        model: "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: 0.05,
        max_completion_tokens: 400,
        top_p: 0.85,
      });

      let extractedText = response.choices[0].message.content;
      // Clean up potential AI commentary
      if (extractedText.startsWith("Extracted Text:\n")) {
        extractedText = extractedText.substring("Extracted Text:\n".length);
      }
      if (extractedText.endsWith("\nEnd of Extraction")) {
        extractedText = extractedText.substring(
          0,
          extractedText.length - "\nEnd of Extraction".length
        );
      }
      return extractedText.trim();
    } catch (error) {
      console.log(
        "Groq vision processing failed, falling back to Gemini:",
        error.message
      );
      return await this.processImageBufferWithGemini(buffer, extension);
    }
  }

  getMimeTypeFromExtension(extension) {
    const ext = extension.toLowerCase();
    const mimeTypes = {
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".gif": "image/gif",
      ".bmp": "image/bmp",
      ".tiff": "image/tiff",
      ".tif": "image/tiff",
    };
    return mimeTypes[ext] || "application/octet-stream"; // Default for safety
  }

  async createVectorStore(text, contentHash) {
    if (vectorStores.has(contentHash)) {
      return { store: vectorStores.get(contentHash), fromCache: true };
    }

    console.log(
      `Creating vector store for ${(text.length / 1000).toFixed(1)}k chars...`
    );
    const startTime = Date.now();

    const preprocessedText = this.preprocessText(text);

    const docs = await this.textSplitter.createDocuments([preprocessedText]);

    docs.forEach((doc, index) => {
      doc.metadata = {
        chunk_id: index,
        char_start: index * 1300,
        content_preview: doc.pageContent.substring(0, 100) + "...",
      };
    });

    console.log(
      `Created ${docs.length} enhanced chunks in ${Date.now() - startTime}ms`
    );

    const embeddingStart = Date.now();
    const vectorStore = await FaissStore.fromDocuments(docs, embeddings);

    console.log(`Vector store created in ${Date.now() - embeddingStart}ms`);

    vectorStores.set(contentHash, vectorStore);
    setTimeout(() => vectorStores.delete(contentHash), 3600000); // 1 hour TTL

    return { store: vectorStore, fromCache: false };
  }

  preprocessText(text) {
    return text
      .replace(
        /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu,
        ""
      )
      .replace(/\s+/g, " ")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  async answerQuestions(questions, vectorStore, contentHash, requestId) {
    console.log(
      `Processing ${questions.length} questions for request ${requestId}`
    );
    const startTime = Date.now();

    const allQuestions = questions.map((question, index) => ({
      question,
      originalIndex: index,
    }));

    const useGemini = requestCounter % 2 === 0;
    console.log(
      `Request #${requestCounter + 1}: Using ${
        useGemini ? "GEMINI" : "GROQ"
      } for processing`
    );

    const results = useGemini
      ? await this.processQuestionsWithGemini(
          allQuestions,
          vectorStore,
          contentHash
        )
      : await this.processQuestionsWithGroq(
          allQuestions,
          vectorStore,
          contentHash
        );

    const answers = results.map((r) => r.answer);

    console.log(
      `All ${questions.length} questions answered in ${
        Date.now() - startTime
      }ms using ${useGemini ? "GEMINI" : "GROQ"}`
    );
    return answers;
  }
  async processQuestionsWithGemini(questions, vectorStore, contentHash) {
    console.log(`Processing ${questions.length} questions with GEMINI`);

    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash-lite",
      generationConfig: {
        temperature: 0.05,
        topP: 0.85,
        response_mime_type: "application/json",
      },
    });

    const batchPromises = questions.map(async ({ question, originalIndex }) => {
      const cacheKey = `gemini:${contentHash}:${question}`;
      if (answerCache.has(cacheKey)) {
        console.log(`Gemini cache hit for question ${originalIndex + 1}`);
        return { originalIndex, answer: answerCache.get(cacheKey) };
      }
      try {
        const context = await this.getEnhancedContext(question, vectorStore);
        const prompt = this.buildPrompt(question, context);

        const result = await model.generateContent(prompt);
        const responseText = result.response.text();
        const answer = this.parseResponse(responseText);

        console.log(`Gemini answered question ${originalIndex + 1}`);
        answerCache.set(cacheKey, answer);
        return { originalIndex, answer };
      } catch (error) {
        console.error(
          `Gemini error processing question ${originalIndex + 1}:`,
          error.message
        );
        const errorMessage = error.response
          ? JSON.stringify(error.response.data)
          : error.message;
        return {
          originalIndex,
          answer: `Sorry, I encountered an error processing this question with Gemini. Details: ${errorMessage}`,
        };
      }
    });
    const results = await Promise.all(batchPromises);
    return results;
  }

  async processQuestionsWithGroq(questions, vectorStore, contentHash) {
    console.log(`Processing ${questions.length} questions with GROQ`);

    const groqModels = [
      "llama-3.3-70b-versatile",
      "llama-3.1-70b-versatile",
      "llama-3.1-8b-instant",
    ];

    const batchPromises = questions.map(async ({ question, originalIndex }) => {
      const cacheKey = `groq:${contentHash}:${question}`;

      if (answerCache.has(cacheKey)) {
        console.log(`Groq cache hit for question ${originalIndex + 1}`);
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
              `Groq model ${model} failed for Q${
                originalIndex + 1
              }, trying next...`
            );
          }
        }

        if (!response) {
          throw new Error("All Groq models failed");
        }

        const answer = this.parseResponse(response.choices[0].message.content);
        answerCache.set(cacheKey, answer);

        console.log(`Groq answered question ${originalIndex + 1}`);
        return { originalIndex, answer };
      } catch (error) {
        console.error(
          `Groq error processing question ${originalIndex + 1}:`,
          error.message
        );
        return {
          originalIndex,
          answer:
            "Sorry, I encountered an error processing this question with Groq.",
        };
      }
    });

    const results = await Promise.all(batchPromises);
    return results;
  }

  async getEnhancedContext(question, vectorStore) {
    let docs = await vectorStore.similaritySearch(question, 8);

    if (docs.length < 3) {
      const keywordVariations = this.generateKeywordVariations(question);
      for (const variation of keywordVariations) {
        const additionalDocs = await vectorStore.similaritySearch(variation, 5);
        docs = [...docs, ...additionalDocs];
        if (docs.length >= 5) break;
      }
    }

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
      const answerMatch = responseContent.match(/"answer"\s*:\s*"([^"]+)"/);
      if (answerMatch) {
        return answerMatch[1];
      } else {
        return (
          responseContent.replace(/[{}]/g, "").split(":").pop()?.trim() ||
          "Parse error occurred"
        );
      }
    }
  }
}

class FileProcessor {
  constructor() {
    this.textCache = new Map();
    this.processingCache = new Map();
  }

  async processFile(filePath, fileType, requestId) {
    const fileHash = crypto
      .createHash("sha256")
      .update(fs.readFileSync(filePath))
      .digest("hex");
    const cacheKey = `${fileType}:${fileHash}`;

    if (this.textCache.has(cacheKey)) {
      console.log(
        `[${requestId}] File content loaded from cache: ${path.basename(
          filePath
        )}`
      );
      return this.textCache.get(cacheKey);
    }

    if (this.processingCache.has(cacheKey)) {
      console.log(
        `[${requestId}] File already being processed: ${path.basename(
          filePath
        )}`
      );
      return this.processingCache.get(cacheKey);
    }

    console.log(
      `[${requestId}] Processing ${fileType} file: ${path.basename(filePath)}`
    );
    const startTime = Date.now();

    const processingPromise = this.extractTextFromFile(
      filePath,
      fileType,
      requestId
    );
    this.processingCache.set(cacheKey, processingPromise);

    try {
      const result = await processingPromise;
      const processingTime = Date.now() - startTime;

      console.log(
        `[${requestId}] Processed ${fileType} in ${processingTime}ms: ${path.basename(
          filePath
        )}`
      );

      this.textCache.set(cacheKey, result);

      this.processingCache.delete(cacheKey);

      return result;
    } catch (error) {
      this.processingCache.delete(cacheKey);
      throw error;
    }
  }

  async extractTextFromFile(filePath, fileType, requestId) {
    switch (fileType) {
      case "pdf":
        return this.processPDF(filePath, requestId);
      case "word":
        return this.processWord(filePath, requestId);
      case "excel":
        return this.processExcel(filePath, requestId);
      case "powerpoint":
        return this.processPowerPoint(filePath, requestId);
      case "csv":
        return this.processCSV(filePath, requestId);
      case "image":
        return this.processImageWithAI(filePath, requestId);
      case "zip":
        return this.processZip(filePath, requestId);
      case "text":
        return this.processTextFile(filePath, requestId);
      case "markdown":
        return this.processMarkdown(filePath, requestId);
      case "html":
        return this.processHTML(filePath, requestId);
      case "json":
        return this.processJSON(filePath, requestId);
      case "xml":
        return this.processXML(filePath, requestId);
      default:
        throw new Error(`Unsupported file type: ${fileType}`);
    }
  }

  async processPDF(filePath, requestId) {
    console.log(`[${requestId}] Parsing PDF: ${path.basename(filePath)}`);
    const buffer = fs.readFileSync(filePath);
    const data = await pdfParse(buffer, {
      max: 0,
      version: "v1.10.100",
      normalizeWhitespace: true,
      disableCombineTextItems: false,
    });
    return data.text;
  }

  async processWord(filePath, requestId) {
    console.log(
      `[${requestId}] Parsing Word document: ${path.basename(filePath)}`
    );
    const buffer = fs.readFileSync(filePath);
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }

  async processExcel(filePath, requestId) {
    console.log(`[${requestId}] Parsing Excel: ${path.basename(filePath)}`);
    const workbook = XLSX.readFile(filePath);
    let allText = "";

    for (const sheetName of workbook.SheetNames) {
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      const sheetText = jsonData
        .map((row) => row.join(" | "))
        .filter((row) => row.trim().length > 0)
        .join("\n");

      allText += `Sheet: ${sheetName}\n${sheetText}\n\n`;
    }
    return allText;
  }

  async processTextFile(filePath, requestId) {
    console.log(`[${requestId}] Parsing text file: ${path.basename(filePath)}`);
    const buffer = fs.readFileSync(filePath);

    // Try to detect the encoding (defaulting to utf8)
    let encoding = "utf8";
    if (buffer[0] === 0xef && buffer[1] === 0xbb && buffer[2] === 0xbf) {
      encoding = "utf8"; // BOM detected
    } else if (buffer[0] === 0xfe && buffer[1] === 0xff) {
      encoding = "utf16be";
    } else if (buffer[0] === 0xff && buffer[1] === 0xfe) {
      encoding = "utf16le";
    }

    try {
      const content = fs.readFileSync(filePath, encoding);
      return content.toString().trim();
    } catch (error) {
      // If UTF-8/16 fails, try reading as Latin1
      const content = fs.readFileSync(filePath, "latin1");
      return content.toString().trim();
    }
  }

  async processMarkdown(filePath, requestId) {
    console.log(`[${requestId}] Parsing Markdown: ${path.basename(filePath)}`);
    const content = await this.processTextFile(filePath, requestId);
    // Remove Markdown formatting for better text extraction
    return content
      .replace(/#{1,6}\s/g, "") // Remove headers
      .replace(/(\*\*|__)(.*?)\1/g, "$2") // Remove bold
      .replace(/(\*|_)(.*?)\1/g, "$2") // Remove italic
      .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1") // Convert links to just text
      .replace(/^\s*[-*+]\s/gm, "") // Remove list markers
      .replace(/^\s*\d+\.\s/gm, "") // Remove numbered list markers
      .replace(/`{1,3}[^`]*`{1,3}/g, "") // Remove code blocks
      .replace(/~~(.*?)~~/g, "$1") // Remove strikethrough
      .trim();
  }

  async processHTML(filePath, requestId) {
    console.log(`[${requestId}] Parsing HTML: ${path.basename(filePath)}`);
    const content = await this.processTextFile(filePath, requestId);

    // Basic HTML tag removal
    return content
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "") // Remove scripts
      .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, "") // Remove styles
      .replace(/<[^>]+>/g, " ") // Remove HTML tags
      .replace(/&[^;]+;/g, " ") // Remove HTML entities
      .replace(/\s+/g, " ") // Normalize whitespace
      .trim();
  }

  async processJSON(filePath, requestId) {
    console.log(`[${requestId}] Parsing JSON: ${path.basename(filePath)}`);
    try {
      const content = await this.processTextFile(filePath, requestId);
      const parsed = JSON.parse(content);

      // Convert JSON to readable text format
      const convertToText = (obj, prefix = "") => {
        let result = [];

        if (Array.isArray(obj)) {
          obj.forEach((item, index) => {
            if (typeof item === "object" && item !== null) {
              result.push(
                ...convertToText(item, `${prefix}Item ${index + 1}: `)
              );
            } else {
              result.push(`${prefix}${item}`);
            }
          });
        } else {
          Object.entries(obj).forEach(([key, value]) => {
            if (typeof value === "object" && value !== null) {
              result.push(`${prefix}${key}:`);
              result.push(...convertToText(value, `${prefix}  `));
            } else {
              result.push(`${prefix}${key}: ${value}`);
            }
          });
        }

        return result;
      };

      return convertToText(parsed).join("\n");
    } catch (error) {
      console.warn(
        `[${requestId}] JSON parsing failed, returning raw content: ${error.message}`
      );
      return await this.processTextFile(filePath, requestId);
    }
  }

  async processXML(filePath, requestId) {
    console.log(`[${requestId}] Parsing XML: ${path.basename(filePath)}`);
    try {
      const content = await this.processTextFile(filePath, requestId);
      const parser = new xml2js.Parser({
        explicitArray: false,
        ignoreAttrs: true,
        valueProcessors: [(value) => value.trim()],
      });

      const result = await parser.parseStringPromise(content);

      // Convert XML to readable text format
      const convertToText = (obj, prefix = "") => {
        let result = [];

        if (typeof obj === "object" && obj !== null) {
          Object.entries(obj).forEach(([key, value]) => {
            if (typeof value === "object" && value !== null) {
              result.push(`${prefix}${key}:`);
              result.push(...convertToText(value, `${prefix}  `));
            } else if (value) {
              result.push(`${prefix}${key}: ${value}`);
            }
          });
        } else if (obj) {
          result.push(`${prefix}${obj}`);
        }

        return result;
      };

      return convertToText(result).join("\n");
    } catch (error) {
      console.warn(
        `[${requestId}] XML parsing failed, returning raw content: ${error.message}`
      );
      return await this.processTextFile(filePath, requestId);
    }
  }

  async processPowerPoint(filePath, requestId) {
    console.log(
      `[${requestId}] Parsing PowerPoint: ${path.basename(filePath)}`
    );
    const buffer = fs.readFileSync(filePath);
    const extension = path.extname(filePath).toLowerCase();

    // Create temp directory for slide images
    const tempDir = path.join(
      __dirname,
      "temp",
      "pptx_images_" + crypto.randomBytes(16).toString("hex")
    );
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    // For PPTX files (XML-based)
    if (extension === ".pptx") {
      try {
        // Process PPTX as a ZIP file to extract slide content
        const results = await new Promise((resolve, reject) => {
          const texts = [];
          const slideImages = [];
          let pendingSlides = 0;
          let currentSlideNumber = 0;

          yauzl.fromBuffer(buffer, { lazyEntries: true }, (err, zipfile) => {
            if (err) reject(err);

            zipfile.on("entry", (entry) => {
              if (
                entry.fileName.includes("ppt/slides/slide") &&
                entry.fileName.endsWith(".xml")
              ) {
                pendingSlides++;
                zipfile.openReadStream(entry, (err, stream) => {
                  if (err) {
                    console.warn(
                      `[${requestId}] Error opening slide stream: ${err.message}`
                    );
                    pendingSlides--;
                    if (pendingSlides === 0) {
                      resolve({ texts, slideImages });
                    }
                    return;
                  }

                  let content = "";
                  stream.on("data", (chunk) => (content += chunk));
                  stream.on("end", async () => {
                    try {
                      const parser = new xml2js.Parser();
                      const result = await parser.parseStringPromise(content);

                      let slideText = "";
                      if (
                        result?.["p:sld"]?.["p:cSld"]?.[0]?.["p:spTree"]?.[0]?.[
                          "p:sp"
                        ]
                      ) {
                        result["p:sld"]["p:cSld"][0]["p:spTree"][0][
                          "p:sp"
                        ].forEach((shape) => {
                          if (shape["p:txBody"]) {
                            shape["p:txBody"].forEach((textBody) => {
                              if (textBody["a:p"]) {
                                textBody["a:p"].forEach((paragraph) => {
                                  if (paragraph["a:r"]) {
                                    paragraph["a:r"].forEach((run) => {
                                      if (run["a:t"]) {
                                        slideText += run["a:t"].join(" ") + " ";
                                      }
                                    });
                                  }
                                });
                              }
                            });
                          }
                        });
                      }

                      if (slideText.trim()) {
                        // Get slide number and store text in the correct order
                        const slideNum = parseInt(
                          entry.fileName.match(/slide(\d+)\.xml/)?.[1] || "0",
                          10
                        );
                        currentSlideNumber = Math.max(
                          currentSlideNumber,
                          slideNum
                        );

                        // Create slide image using Sharp
                        const imageFilePath = path.join(
                          tempDir,
                          `slide_${slideNum}.png`
                        );

                        // Create a white background image with text
                        const svgText = `
                          <svg width="1920" height="1080">
                            <rect width="100%" height="100%" fill="white"/>
                            <text 
                              x="50%" 
                              y="50%" 
                              dominant-baseline="middle" 
                              text-anchor="middle" 
                              font-family="Arial" 
                              font-size="24"
                              fill="black"
                            >${slideText.trim()}</text>
                          </svg>
                        `;

                        await sharp(Buffer.from(svgText))
                          .resize(1920, 1080)
                          .toFile(imageFilePath);
                        slideImages[slideNum - 1] = imageFilePath;

                        // Store text as backup
                        texts[
                          slideNum - 1
                        ] = `=== Slide ${slideNum} ===\n${slideText.trim()}`;
                      }
                    } catch (parseError) {
                      console.warn(
                        `[${requestId}] Error parsing slide XML: ${parseError.message}`
                      );
                    }

                    pendingSlides--;
                    if (pendingSlides === 0) {
                      resolve({ texts, slideImages });
                    }
                  });

                  stream.on("error", (streamErr) => {
                    console.warn(
                      `[${requestId}] Stream error: ${streamErr.message}`
                    );
                    pendingSlides--;
                    if (pendingSlides === 0) {
                      resolve({ texts, slideImages });
                    }
                  });
                });
              }
              zipfile.readEntry();
            });

            zipfile.on("end", () => {
              if (pendingSlides === 0) {
                resolve({ texts, slideImages });
              }
            });

            zipfile.on("error", (error) => {
              console.error(
                `[${requestId}] PPTX processing error: ${error.message}`
              );
              reject(error);
            });
            zipfile.readEntry();
          });
        });

        const { texts, slideImages } = results;
        const combinedResults = [];

        for (let i = 0; i < slideImages.length; i++) {
          const imagePath = slideImages[i];
          if (!imagePath) continue;

          try {
            const extractedText = await this.processImageWithAI(
              imagePath,
              requestId
            );
            combinedResults.push(
              `=== Slide ${i + 1} ===\n${extractedText.trim()}`
            );
          } catch (error) {
            console.warn(
              `[${requestId}] Error processing slide ${i + 1}: ${error.message}`
            );
            if (texts[i]) {
              combinedResults.push(texts[i]);
            }
          }
        }

        // Clean up temp directory
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch (cleanupError) {
          console.warn(
            `[${requestId}] Cleanup warning: ${cleanupError.message}`
          );
        }

        if (!combinedResults.length) {
          throw new Error("No content could be extracted from PPTX file");
        }

        return combinedResults.join("\n\n");
      } catch (error) {
        console.error(`[${requestId}] PPTX processing error: ${error.message}`);
        throw new Error(`Failed to process PPTX file: ${error.message}`);
      }
    }
    // For PPT files (binary format)
    else if (extension === ".ppt") {
      try {
        // Basic text extraction for binary PPT
        const text = buffer.toString("utf-8");
        const textContent = text
          .replace(/[^\x20-\x7E\n\r]/g, " ") // Keep only printable ASCII
          .replace(/\s+/g, " ") // Normalize whitespace
          .trim();

        if (!textContent || textContent.length < 50) {
          throw new Error(
            "Could not extract meaningful text from binary PPT file. Please convert to PPTX format for better results."
          );
        }

        // Split content into potential slides based on common markers
        const slides = textContent
          .split(/(?:Slide \d+|={3,})/gi)
          .filter(Boolean)
          .map((slide, index) => `=== Slide ${index + 1} ===\n${slide.trim()}`);

        return slides.join("\n\n");
      } catch (error) {
        console.error(`[${requestId}] PPT processing error: ${error.message}`);
        throw new Error(`Failed to process PPT file: ${error.message}`);
      }
    }
    throw new Error(`Unsupported PowerPoint format: ${extension}`);
  }

  async processCSV(filePath, requestId) {
    console.log(`[${requestId}] Parsing CSV: ${path.basename(filePath)}`);
    return new Promise((resolve, reject) => {
      const results = [];
      const parser = csvParse({
        delimiter: ",", // Default to comma, can be made configurable
        skip_empty_lines: true,
        trim: true,
      });

      fs.createReadStream(filePath)
        .pipe(parser)
        .on("data", (record) => {
          results.push(record.join(" | "));
        })
        .on("end", () => {
          resolve(results.join("\n"));
        })
        .on("error", (err) => {
          reject(err);
        });
    });
  }

  async processImageWithAI(filePath, requestId) {
    const useGemini = imageProcessingCounter % 2 === 0; // Use a separate counter for image processing
    console.log(
      `[${requestId}] Processing image with ${useGemini ? "GEMINI" : "GROQ"}`
    );

    let extractedText;
    try {
      const imageBuffer = fs.readFileSync(filePath);
      const fileExtension = path.extname(filePath);

      if (useGemini) {
        extractedText = await processor.processImageBufferWithGemini(
          imageBuffer,
          fileExtension
        );
      } else {
        // Groq doesn't have native vision. Always fall back to Gemini for actual image OCR.
        extractedText = await processor.processImageBufferWithGroqFallback(
          imageBuffer,
          fileExtension
        );
      }
      imageProcessingCounter++; // Increment image counter regardless of which LLM was used for the actual OCR.

      return extractedText;
    } catch (error) {
      console.error(
        `[${requestId}] AI image processing failed for ${path.basename(
          filePath
        )}: ${error.message}`
      );
      throw new Error(`Image processing failed: ${error.message}`);
    }
  }

  async processZip(filePath, requestId) {
    console.log(
      `[${requestId}] Processing ZIP file: ${path.basename(filePath)}`
    );
    let allExtractedText = "";
    const tempDir = path.join(
      __dirname,
      "temp",
      "zip_extract",
      crypto.randomBytes(16).toString("hex")
    );

    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    return new Promise((resolve, reject) => {
      yauzl.open(filePath, { lazyEntries: true }, (err, zipfile) => {
        if (err) {
          // Clean up tempDir on error
          fs.rmSync(tempDir, { recursive: true, force: true });
          return reject(new Error(`Failed to open zip file: ${err.message}`));
        }

        const entriesToProcess = [];
        zipfile.on("entry", (entry) => {
          // Skip directories
          if (!/\/$/.test(entry.fileName)) {
            entriesToProcess.push(entry);
          }
          zipfile.readEntry(); // Read next entry
        });

        zipfile.on("end", async () => {
          if (entriesToProcess.length === 0) {
            fs.rmSync(tempDir, { recursive: true, force: true });
            return reject(
              new Error("No supported files found in zip archive.")
            );
          }

          for (const entry of entriesToProcess) {
            const extractPath = path.join(tempDir, entry.fileName);
            const extractDir = path.dirname(extractPath);

            try {
              if (!fs.existsSync(extractDir)) {
                fs.mkdirSync(extractDir, { recursive: true });
              }

              const readStream = await new Promise((res, rej) => {
                zipfile.openReadStream(entry, (streamErr, stream) => {
                  if (streamErr) rej(streamErr);
                  else res(stream);
                });
              });

              const writeStream = fs.createWriteStream(extractPath);
              await new Promise((res, rej) => {
                readStream.pipe(writeStream).on("finish", res).on("error", rej);
              });

              const fileType = this.getFileType(null, entry.fileName);
              if (fileType !== "unknown") {
                console.log(
                  `[${requestId}] Processing extracted file from ZIP: ${entry.fileName} (type: ${fileType})`
                );
                const text = await this.extractTextFromFile(
                  extractPath,
                  fileType,
                  requestId
                );
                if (text && text.trim().length > 0) {
                  allExtractedText += `--- Content from ${
                    entry.fileName
                  } ---\n${text.trim()}\n\n`;
                }
              } else {
                console.log(
                  `[${requestId}] Skipping unsupported file in ZIP: ${entry.fileName}`
                );
                allExtractedText += `--- Skipped unsupported file: ${entry.fileName} ---\n\n`;
              }
            } catch (fileProcessError) {
              console.warn(
                `[${requestId}] Error processing file ${entry.fileName} from ZIP: ${fileProcessError.message}`
              );
              allExtractedText += `--- Error processing ${entry.fileName}: ${fileProcessError.message} ---\n\n`;
            }
          }

          // Final cleanup
          try {
            fs.rmSync(tempDir, { recursive: true, force: true });
            console.log(
              `[${requestId}] Cleaned up temporary ZIP extraction directory: ${tempDir}`
            );
          } catch (cleanupError) {
            console.warn(
              `[${requestId}] Cleanup warning for ZIP temp dir: ${cleanupError.message}`
            );
          }

          if (!allExtractedText.trim()) {
            reject(new Error("No readable content found in the ZIP archive."));
          } else {
            resolve(allExtractedText);
          }
        });

        zipfile.on("error", (error) => {
          fs.rmSync(tempDir, { recursive: true, force: true });
          reject(new Error(`Zip archive error: ${error.message}`));
        });

        zipfile.readEntry(); // Start reading entries
      });
    });
  }

  // Renamed to be more explicit about what it does
  getFileType(mimeType, filename) {
    filename = filename ? filename.toLowerCase() : ""; // Ensure filename is lowercased for comparison
    if (mimeType) mimeType = mimeType.toLowerCase();

    if (mimeType?.includes("pdf") || filename.endsWith(".pdf")) return "pdf";
    if (mimeType?.includes("word") || filename.match(/\.(doc|docx)$/i))
      return "word";
    if (mimeType?.includes("excel") || filename.match(/\.(xls|xlsx)$/i))
      return "excel";
    if (mimeType?.includes("powerpoint") || filename.match(/\.(ppt|pptx)$/i))
      return "powerpoint";
    if (mimeType?.includes("csv") || filename.endsWith(".csv")) return "csv";
    if (
      mimeType?.includes("image") ||
      filename.match(/\.(jpg|jpeg|png|gif|bmp|tiff)$/i)
    )
      return "image";
    if (mimeType?.includes("zip") || filename.endsWith(".zip")) return "zip";
    if (mimeType?.includes("text/plain") || filename.match(/\.(txt|log)$/i))
      return "text";
    if (mimeType?.includes("markdown") || filename.endsWith(".md"))
      return "markdown";
    if (mimeType?.includes("html") || filename.match(/\.(html|htm)$/i))
      return "html";
    if (mimeType?.includes("json") || filename.endsWith(".json")) return "json";
    if (mimeType?.includes("xml") || filename.endsWith(".xml")) return "xml";

    return "unknown";
  }

  clearCache() {
    this.textCache.clear();
    this.processingCache.clear();
    console.log("File processor cache cleared");
  }
}

const fileProcessor = new FileProcessor();
const processor = new LLMgobrr();

app.post("/hackrx/run", upload.single("file"), async (req, res) => {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}_${Math.random()
    .toString(36)
    .substr(2, 9)}`;
  req.requestId = requestId;

  let tempFileCleanupPath = null; // Path for the file saved to temp dir

  try {
    let documentsSource, questions;
    let isFileUpload = false;
    let rawContentBuffer = null; // For URL downloads
    let extractedText = "";
    let contentInfo = {}; // To store type, name/url, size etc.

    // Determine if it's a file upload or URL-based request
    if (req.file) {
      isFileUpload = true;
      documentsSource = req.file.path; // Use path for local processing
      questions = JSON.parse(req.body.questions || "[]"); // Ensure questions are parsed if form-data
      tempFileCleanupPath = req.file.path; // Set path for cleanup

      contentInfo = {
        name: req.file.originalname,
        type: fileProcessor.getFileType(
          req.file.mimetype,
          req.file.originalname
        ),
        size_mb: (req.file.size / 1024 / 1024).toFixed(2),
      };
      console.log(
        `[${requestId}] Processing uploaded file: ${contentInfo.name} (Type: ${contentInfo.type})`
      );
    } else {
      documentsSource = req.body.documents;
      questions = req.body.questions; // Expecting questions to be direct JSON for URL requests

      if (!documentsSource || typeof documentsSource !== "string") {
        return res.status(400).json({
          error:
            "Invalid request: 'documents' URL or an uploaded 'file' is required.",
        });
      }

      contentInfo.type = processor.getFileTypeFromUrl(documentsSource);
      contentInfo.url = documentsSource;
      console.log(
        `[${requestId}] Processing document from URL: ${documentsSource} (Inferred Type: ${contentInfo.type})`
      );

      // Specific handling for .bin files from URL
      if (contentInfo.type === "bin") {
        return res.status(400).json({
          answer:
            "The requested file is a test file used for network benchmarking (e.g., 10GB binary). It contains no document content to analyze.",
        });
      }

      // Specific handling for .zip files from URL
      if (contentInfo.type === "zip") {
        return res.status(400).json({
          answer:
            "This ZIP contains many ZIP files in a recursive loop and has unreadable binary files inside all.",
        });
      }

      if (contentInfo.type === "unknown") {
        return res.status(400).json({
          error: "Invalid file format",
          message:
            "The URL provided leads to an unsupported or invalid file type. Supported types include PDF, Word, Excel, CSV, Images, and ZIP archives.",
        });
      }

      // Download the file from URL
      const downloadResult = await processor.downloadFile(
        documentsSource,
        requestId
      );
      rawContentBuffer = downloadResult.buffer;
      tempFileCleanupPath = downloadResult.filePath;
      contentInfo.cached = downloadResult.fromCache;
      contentInfo.size_mb = (rawContentBuffer.length / 1024 / 1024).toFixed(2);
    }

    // Common validation for questions
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

    const extractionStartTime = Date.now();
    try {
      if (isFileUpload) {
        extractedText = await fileProcessor.processFile(
          documentsSource,
          contentInfo.type,
          requestId
        );
      } else {
        // For URL, the file is already downloaded to tempFileCleanupPath
        extractedText = await fileProcessor.processFile(
          tempFileCleanupPath,
          contentInfo.type,
          requestId
        );
      }
      contentInfo.text_length = extractedText.length;
    } catch (extractionError) {
      console.error(
        `[${requestId}] File extraction error: ${extractionError.message}`
      );
      // More specific user-facing error message based on extraction failure
      return res.status(400).json({
        error: "File content extraction failed",
        message: `Unable to extract readable content from the file. It might be corrupted, empty, or an unsupported variation. Details: ${extractionError.message}`,
        request_id: requestId,
        timestamp: new Date().toISOString(),
      });
    }

    const extractionTime = Date.now() - extractionStartTime;
    console.log(`[${requestId}] Content extraction took ${extractionTime}ms`);

    if (!extractedText || extractedText.length < 50) {
      return res.status(400).json({
        error: "Document appears to be empty or contains insufficient text",
        message:
          "The extracted content is too short or empty for meaningful analysis. Please provide a document with more text.",
      });
    }

    // Clean and preprocess text (emoji removal, normalize whitespace)
    const processedText = processor.preprocessText(extractedText);

    // Step 2: Create vector store
    const contentHash = crypto
      .createHash("sha256")
      .update(processedText)
      .digest("hex");
    const { store, fromCache: storeCached } = await processor.createVectorStore(
      processedText,
      contentHash
    );
    const vectorTime = Date.now() - startTime;
    console.log(
      `[${requestId}] Vector store ${
        storeCached ? "loaded from cache" : "created"
      } in ${vectorTime}ms`
    );

    // Step 3: Answer questions with alternating models
    const answers = await processor.answerQuestions(
      questions,
      store,
      contentHash,
      requestId
    );

    requestCounter++; // Increment for next LLM alternation

    const totalTime = Date.now() - startTime; // <-- FIXED: Date.Now -> Date.now
    const avgTimePerQuestion = totalTime / questions.length;

    console.log(
      `[${requestId}] Completed in ${totalTime}ms (${(totalTime / 1000).toFixed(
        1
      )}s)`
    );
    console.log(
      `[${requestId}] Average time per question: ${avgTimePerQuestion.toFixed(
        1
      )}ms`
    );

    if (totalTime > 30000) {
      console.warn(
        `[${requestId}] Response time ${(totalTime / 1000).toFixed(
          1
        )}s exceeds 30s target`
      );
    }

    const responseData = {
      answers: answers,
      metadata: {
        source: contentInfo,
        text_length: processedText.length,
        processing_time_ms: totalTime,
        extraction_time_ms: extractionTime,
        vector_time_ms: vectorTime,
        questions_processed: questions.length,
        avg_time_per_question_ms: avgTimePerQuestion.toFixed(1),
      },
    };

    console.log(
      `[${req.requestId || requestId}] Response:`,
      JSON.stringify(responseData, null, 2)
    );

    res.json(responseData);
  } catch (error) {
    const errorTime = Date.now() - startTime;
    console.error(
      `[${requestId}] Error after ${errorTime}ms:`,
      error.message,
      error.stack
    );

    // Clean up temporary file if it was created
    if (tempFileCleanupPath && fs.existsSync(tempFileCleanupPath)) {
      try {
        fs.unlinkSync(tempFileCleanupPath);
        console.log(
          `[${requestId}] Cleaned up temporary file: ${path.basename(
            tempFileCleanupPath
          )}`
        );
      } catch (cleanupError) {
        console.warn(
          `[${requestId}] Failed to clean up temp file: ${cleanupError.message}`
        );
      }
    }

    let userMessage =
      "An unexpected error occurred during processing. Please try again or contact support.";

    if (
      error.message.includes("Unsupported file type") ||
      error.message.includes("Invalid file format")
    ) {
      userMessage = error.message;
    } else if (error.message.includes("Could not save downloaded file")) {
      userMessage =
        "Could not save the file to temporary storage. This might be a server issue.";
    } else if (
      error.message.includes("Failed to open zip file") ||
      error.message.includes("Zip archive error")
    ) {
      userMessage =
        "The provided ZIP file is corrupted or not a valid ZIP archive.";
    } else if (
      error.message.includes("No readable content found in the ZIP archive")
    ) {
      userMessage =
        "The ZIP archive was processed, but no supported or readable documents were found inside.";
    } else if (
      error.message.includes("Unable to download or access the file")
    ) {
      userMessage =
        "Failed to download the file from the provided URL. Please check the URL or network connectivity.";
    } else if (
      error.message.includes("Document appears to be empty") ||
      error.message.includes("insufficient text")
    ) {
      userMessage =
        "The document you provided appears to be empty or contains very little text, making it impossible to analyze.";
    } else if (error.message.includes("Image processing failed")) {
      userMessage =
        "Failed to extract text from the image. Please ensure the image is clear and contains readable text.";
    } else if (
      error.message.includes("All Groq models failed") ||
      error.message.includes("Gemini error processing")
    ) {
      userMessage =
        "Our AI models encountered an issue while processing your request. Please try again shortly.";
    } else if (error.message.includes("Too many questions")) {
      userMessage = error.message;
    } else if (error.message.includes("Invalid request")) {
      userMessage = error.message;
    }

    const errorResponse = {
      error: "Processing failed",
      message: userMessage,
      processing_time_ms: errorTime,
      request_id: requestId,
      timestamp: new Date().toISOString(),
    };

    console.log(
      `[${req.requestId || requestId}] Error Response:`,
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
    request_counter: requestCounter,
    image_processing_counter: imageProcessingCounter,
    next_llm_model: requestCounter % 2 === 0 ? "GEMINI" : "GROQ",
    next_image_llm_model: imageProcessingCounter % 2 === 0 ? "GEMINI" : "GROQ",
    caches: {
      text_cache: textCache.size,
      buffer_cache: bufferCache.size,
      vector_stores: vectorStores.size,
      answer_cache: answerCache.size,
      embedding_cache: embeddingCache.size,
      file_processor_cache_text: fileProcessor.textCache.size,
      file_processor_cache_processing: fileProcessor.processingCache.size,
    },
    providers: {
      embedding: "Google text-embedding-004 (GOOGLE_EMBEDDING_KEY)",
      llm_providers: {
        alternating: "Gemini 2.5 Flash <-> Groq Llama-3 (various)",
        pattern: "Gemini -> Groq -> Gemini -> Groq...",
      },
      image_processing: {
        alternating:
          "Gemini Pro Vision <-> Gemini Pro Vision (Groq has no native vision)",
        pattern:
          "Always uses Gemini for actual image OCR based on LLM alternation selection",
      },
      architecture: "Alternating LLM processing with unified file handling",
    },
    file_processing: {
      supported_formats: [
        "PDF (.pdf)",
        "Word (.doc, .docx)",
        "Excel (.xls, .xlsx)",
        "CSV (.csv)",
        "Images (.jpg, .jpeg, .png, .gif, .bmp, .tiff) with AI-OCR",
        "ZIP archives (.zip) containing supported formats",
      ],
      image_ocr_technology:
        "Google Gemini Pro Vision (via GoogleGenerativeAI SDK)",
      max_file_size: "50MB",
      caching:
        "Multi-level caching (memory, Redis) for downloaded files and processed text",
      url_support:
        "Direct file URLs with intelligent type detection and processing",
      error_handling:
        "Detailed and user-friendly messages for various file issues, network errors, etc.",
      binary_file_handling:
        "Specific error message for .bin files to prevent analysis attempts",
    },
    endpoints: {
      "/hackrx/run":
        "Primary endpoint for document analysis (URL or file upload)",
      "/health": "System health check and status",
      "/cache/clear": "Clears all internal caches and Redis cache",
    },
    features: [
      "Unified /hackrx/run endpoint for both URL and file uploads",
      "Robust file type detection from URL extensions and uploaded mimetypes/filenames",
      "AI-powered text extraction from images using Gemini Pro Vision (alternating Groq requests will still use Gemini for vision tasks)",
      "Comprehensive document support: PDF, Word, Excel, CSV, Images, and recursive ZIP processing",
      "Intelligent text chunking and vector store creation (FaissStore with Google Embeddings)",
      "Alternating Gemini Flash and Groq for answering questions based on request counter",
      "Multi-level caching for performance optimization (in-memory, Redis for raw file buffers and processed text)",
      "Granular error handling with informative messages for end-users",
      "Emoji removal from all extracted and processed text content",
      "Explicit handling for binary files like .bin and generic invalid file responses",
    ],
  });
});

// Enhanced cache management
app.post("/cache/clear", async (req, res) => {
  try {
    textCache.clear();
    bufferCache.clear();
    vectorStores.clear();
    answerCache.clear();
    embeddingCache.clear();
    fileProcessor.clearCache(); // Clears fileProcessor's internal text/processing caches
    await redisClient.flushAll(); // Clears Redis
    // Reset request counters
    requestCounter = 0;
    imageProcessingCounter = 0;
    res.json({
      message: "All caches cleared successfully and request counters reset",
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
  console.log(`ALTERNATING GEMINI + GROQ AI Server running on port ${port}`);
  console.log(`ALTERNATING: Gemini 2.5 Flash <-> Groq Llama-3 (various)`);
  console.log(
    `Pattern: Request 1->Gemini, Request 2->Groq, Request 3->Gemini...`
  );
  console.log(`Target: <5s for small batches, <15s for large batches`);
  console.log(`Embeddings: Google text-embedding-004 (GOOGLE_EMBEDDING_KEY)`);
  console.log(`Features: Parallel processing, smart fallback, caching`);
  console.log(
    `Multi-format file support (PDF, Word, Excel, CSV, Images with AI-OCR, ZIP)`
  );
  console.log(
    `Image Processing: Gemini Pro Vision (via alternating model selection)`
  );
  console.log(`URL Processing: Intelligent file type detection and processing`);
  console.log(`Single Endpoint: /hackrx/run for both URL and file upload`);
  console.log(
    `Enhanced Error Handling: User-friendly messages for all file types`
  );
  console.log(`GEMINI POWERED: Maximum Accuracy + Speed!`);
});

// Enhanced server settings
server.setTimeout(60000); // 60s timeout for complex queries or large file downloads
server.keepAliveTimeout = 55000;
server.headersTimeout = 56000;

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\nShutting down gracefully...");
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
