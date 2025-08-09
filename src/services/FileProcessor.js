const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const XLSX = require("xlsx");
const sharp = require("sharp");
const yauzl = require("yauzl");
const { parse: csvParse } = require("csv-parse");
const xml2js = require("xml2js");
const { getMimeTypeFromExtension } = require("../utils/fileUtils");

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
      case "webpage":
        return this.processWebpage(filePath, requestId);
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

  async processWebpage(filePath, requestId) {
    console.log(
      `[${requestId}] Processing webpage content: ${path.basename(filePath)}`
    );
    try {
      return await this.processTextFile(filePath, requestId);
    } catch (error) {
      console.error(
        `[${requestId}] Webpage processing error: ${error.message}`
      );
      throw error;
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

    let encoding = "utf8";
    if (buffer[0] === 0xef && buffer[1] === 0xbb && buffer[2] === 0xbf) {
      encoding = "utf8";
    } else if (buffer[0] === 0xfe && buffer[1] === 0xff) {
      encoding = "utf16be";
    } else if (buffer[0] === 0xff && buffer[1] === 0xfe) {
      encoding = "utf16le";
    }

    try {
      const content = fs.readFileSync(filePath, encoding);
      return content.toString().trim();
    } catch (error) {
      const content = fs.readFileSync(filePath, "latin1");
      return content.toString().trim();
    }
  }

  async processMarkdown(filePath, requestId) {
    console.log(`[${requestId}] Parsing Markdown: ${path.basename(filePath)}`);
    const content = await this.processTextFile(filePath, requestId);
    return content
      .replace(/#{1,6}\s/g, "")
      .replace(/(\*\*|__)(.*?)\1/g, "$2")
      .replace(/(\*|_)(.*?)\1/g, "$2")
      .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
      .replace(/^\s*[-*+]\s/gm, "")
      .replace(/^\s*\d+\.\s/gm, "")
      .replace(/`{1,3}[^`]*`{1,3}/g, "")
      .replace(/~~(.*?)~~/g, "$1")
      .trim();
  }

  async processHTML(filePath, requestId) {
    console.log(`[${requestId}] Parsing HTML: ${path.basename(filePath)}`);
    const content = await this.processTextFile(filePath, requestId);

    return content
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "")
      .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/&[^;]+;/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  async processJSON(filePath, requestId) {
    console.log(`[${requestId}] Parsing JSON: ${path.basename(filePath)}`);
    try {
      const content = await this.processTextFile(filePath, requestId);
      const parsed = JSON.parse(content);

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

    const tempDir = path.join(
      path.dirname(filePath),
      "pptx_images_" + crypto.randomBytes(16).toString("hex")
    );
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    if (extension === ".pptx") {
      try {
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
                        const slideNum = parseInt(
                          entry.fileName.match(/slide(\d+)\.xml/)?.[1] || "0",
                          10
                        );
                        currentSlideNumber = Math.max(
                          currentSlideNumber,
                          slideNum
                        );

                        const imageFilePath = path.join(
                          tempDir,
                          `slide_${slideNum}.png`
                        );

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
    } else if (extension === ".ppt") {
      try {
        const text = buffer.toString("utf-8");
        const textContent = text
          .replace(/[^\x20-\x7E\n\r]/g, " ")
          .replace(/\s+/g, " ")
          .trim();

        if (!textContent || textContent.length < 50) {
          throw new Error(
            "Could not extract meaningful text from binary PPT file. Please convert to PPTX format for better results."
          );
        }

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
        delimiter: ",",
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
    const {
      imageProcessingCounter,
      incrementImageCounter,
    } = require("../utils/cache");
    const LLMService = require("./LLMService");
    const llmService = new LLMService();

    const useGemini = imageProcessingCounter % 2 === 0;
    console.log(
      `[${requestId}] Processing image with ${useGemini ? "GEMINI" : "GROQ"}`
    );

    let extractedText;
    try {
      const imageBuffer = fs.readFileSync(filePath);
      const fileExtension = path.extname(filePath);

      if (useGemini) {
        extractedText = await llmService.processImageBufferWithGemini(
          imageBuffer,
          fileExtension
        );
      } else {
        extractedText = await llmService.processImageBufferWithGroqFallback(
          imageBuffer,
          fileExtension
        );
      }
      incrementImageCounter();

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
      path.dirname(filePath),
      "zip_extract",
      crypto.randomBytes(16).toString("hex")
    );

    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    return new Promise((resolve, reject) => {
      yauzl.open(filePath, { lazyEntries: true }, (err, zipfile) => {
        if (err) {
          fs.rmSync(tempDir, { recursive: true, force: true });
          return reject(new Error(`Failed to open zip file: ${err.message}`));
        }

        const entriesToProcess = [];
        zipfile.on("entry", (entry) => {
          if (!/\/$/.test(entry.fileName)) {
            entriesToProcess.push(entry);
          }
          zipfile.readEntry();
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

        zipfile.readEntry();
      });
    });
  }

  getFileType(mimeType, filename) {
    filename = filename ? filename.toLowerCase() : "";
    if (mimeType) mimeType = mimeType.toLowerCase();

    if (filename.startsWith("scraped_")) {
      return "webpage";
    }

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
    if (!mimeType && !filename) {
      return "webpage";
    }

    return "unknown";
  }

  clearCache() {
    this.textCache.clear();
    this.processingCache.clear();
    console.log("File processor cache cleared");
  }
}

module.exports = FileProcessor;
