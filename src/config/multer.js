const multer = require("multer");
const path = require("path");
const fs = require("fs");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "..", "..", "temp", "uploads");
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
          /\.(pdf|doc|docx|xls|xlsx|ppt|pptx|csv|jpg|jpeg|png|gif|bmp|tiff|zip|txt|md|html|json|xml)$/i
        )
    ) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Unsupported file type: ${file.mimetype}. Allowed types include PDF, Word, Excel, CSV, Image, Text, Markdown, HTML, JSON, XML, and ZIP.`
        ),
        false
      );
    }
  },
});

module.exports = upload;
