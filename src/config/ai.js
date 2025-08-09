require("dotenv").config({
  path: require("path").join(__dirname, "../../.env"),
});

const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { Groq } = require("groq-sdk");

// Initialize AI services only if API keys are available
let groq = null;
let genAI = null;
let embeddings = null;

try {
  if (process.env.GROQ_API_KEY) {
    groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    console.log("✅ Groq service initialized");
  } else {
    console.warn("⚠️  GROQ_API_KEY not set, Groq service disabled");
  }
} catch (error) {
  console.warn("⚠️  Failed to initialize Groq service:", error.message);
}

try {
  if (process.env.GOOGLE_API_KEY) {
    genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    console.log("✅ Google GenerativeAI service initialized");
  } else {
    console.warn(
      "⚠️  GOOGLE_API_KEY not set, Google GenerativeAI service disabled"
    );
  }
} catch (error) {
  console.warn(
    "⚠️  Failed to initialize Google GenerativeAI service:",
    error.message
  );
}

try {
  if (process.env.GOOGLE_API_KEY) {
    embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_EMBEDDING_KEY,
      model: "text-embedding-004",
    });
    console.log("✅ Google Embeddings service initialized");
  } else {
    console.warn(
      "⚠️  GOOGLE_API_KEY not set, Google Embeddings service disabled"
    );
  }
} catch (error) {
  console.warn(
    "⚠️  Failed to initialize Google Embeddings service:",
    error.message
  );
}

module.exports = {
  groq,
  genAI,
  embeddings,
};
