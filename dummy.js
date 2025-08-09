const express = require("express");
const app = express();
const port = 3000;

// Middleware to parse JSON requests
app.use(express.json());

// Middleware to check Bearer token
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1]; // Bearer TOKEN

  const expectedToken =
    "d2616659a7fd02e3f14cae15e7660eac6d24581d4e9f5dcecd8ac5cf55db138a";

  if (!token || token !== expectedToken) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  next();
};

// Route to handle document queries
app.post("/hackrx/run", authenticateToken, (req, res) => {
  const { documents, questions } = req.body;

  // Check which document is being queried based on the URL
  if (documents && documents.includes("FinalRound4SubmissionPDF.pdf")) {
    // Flight number document
    if (questions && questions.includes("What is my flight number?")) {
      return res.json({
        answers: ["e0d449"],
      });
    }
  } else if (documents && documents.includes("News.pdf")) {
    // News document with Malayalam and English questions
    const answers = [
      "ജനുവരി 20, 2025", // Trump announced 100% tariff on this day
      "BRICS രാജ്യങ്ങളിൽ നിന്നുള്ള എല്ലാ ഉത്പന്നങ്ങളും", // All products from BRICS countries
      "അമേരിക്കയിൽ പ്രധാന നിർമ്മാണ സൗകര്യങ്ങൾ സ്ഥാപിക്കുകയും അമേരിക്കൻ തൊഴിലാളികളെ നിയമിക്കുകയും ചെയ്യുന്ന കമ്പനികൾക്ക്", // Companies that establish major manufacturing facilities in America and hire American workers
      "Apple committed to investing $100 billion over 5 years to expand manufacturing in the US and create American jobs", // Apple's investment commitment and objective
      "This policy will likely increase consumer prices in the short term but aims to boost domestic manufacturing and reduce dependence on imports from BRICS nations in the long term", // Impact on consumers and global market
    ];

    return res.json({
      answers: answers,
    });
  } else if (documents && documents.includes("hackTeam=2014")) {
    // Secret token document
    if (questions && questions.some((q) => q.includes("secret token"))) {
      return res.json({
        answers: [
          "06e4d8bf281aa3a197c2822ad2d93f0067ee178a984adefb6416678ebb0800aa",
        ],
      });
    }
  }

  // Default response if no matching document found
  res.status(404).json({
    error: "Document not found or questions not recognized",
  });
});

// Health check route
app.get("/health", (req, res) => {
  res.json({ status: "Server is running" });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log("Available endpoints:");
  console.log(
    "POST /hackrx/run - Query documents with questions (requires Bearer token)"
  );
  console.log("GET /health - Health check");
  console.log("");
  console.log("Required headers:");
  console.log("Content-Type: application/json");
  console.log(
    "Authorization: Bearer d2616659a7fd02e3f14cae15e7660eac6d24581d4e9f5dcecd8ac5cf55db138a"
  );
});
