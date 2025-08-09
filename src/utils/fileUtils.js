const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ensureTempDirectory = () => {
  const tempDir = path.join(__dirname, "..", "..", "temp");
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  return tempDir;
};

const getFileExtensionFromUrl = (url) => {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    const extension = path.extname(pathname).toLowerCase();
    return extension || "";
  } catch (error) {
    return "";
  }
};

const getFileTypeFromUrl = (url) => {
  if (!url.includes(".")) {
    return "webpage";
  }

  if (url.includes("hackrx.in") || url.includes("register.hackrx.in")) {
    return "webpage";
  }

  const extension = getFileExtensionFromUrl(url);
  if (!extension) {
    return "webpage";
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
  if ([".txt", ".log"].includes(extension)) return "text";
  if (extension === ".md") return "markdown";
  if ([".html", ".htm"].includes(extension)) return "html";
  if (extension === ".json") return "json";
  if (extension === ".xml") return "xml";

  return "unknown";
};

const getMimeTypeFromExtension = (extension) => {
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
  return mimeTypes[ext] || "application/octet-stream";
};

const createFileHash = (content) => {
  return crypto.createHash("sha256").update(content).digest("hex");
};

const cleanupTempFile = (filePath) => {
  if (filePath && fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
      return true;
    } catch (error) {
      console.warn(`Failed to clean up temp file: ${error.message}`);
      return false;
    }
  }
  return false;
};

module.exports = {
  ensureTempDirectory,
  getFileExtensionFromUrl,
  getFileTypeFromUrl,
  getMimeTypeFromExtension,
  createFileHash,
  cleanupTempFile
};
