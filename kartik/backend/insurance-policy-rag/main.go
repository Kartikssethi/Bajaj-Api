package main

import (
	"context"
	"crypto/sha256"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"github.com/ledongthuc/pdf"
	"google.golang.org/api/option"
)

// Structures
type EmbeddingCache struct {
	Embedding []float32 `json:"embedding"`
}

type FAISSIndex struct {
	Vectors [][]float32          `json:"vectors"`
	Mapping map[int]string       `json:"mapping"`
}

type AskRequest struct {
	Question string `json:"question"`
}

type HackRxRequest struct {
	Documents string   `json:"documents"`
	Questions []string `json:"questions"`
}

type AnswerResponse struct {
	Answer       string `json:"answer"`
	SourceClause string `json:"source_clause"`
	Reasoning    string `json:"reasoning"`
}

// Global variables
var (
	redisClient *redis.Client
	genaiClient *genai.Client
	faissIndex  *FAISSIndex
	kartikDir   string
)

const (
	EMBEDDING_DIM = 768
)

func init() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found")
	}

	// Initialize Redis
	redisClient = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0,
	})

	// Initialize Gemini AI
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY is not set in .env")
	}

	ctx := context.Background()
	var err error
	genaiClient, err = genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatal("Failed to create genai client:", err)
	}

	// Setup directories
	wd, _ := os.Getwd()
	kartikDir = filepath.Join(filepath.Dir(wd), "kartik")
	os.MkdirAll(kartikDir, 0755)

	// Initialize FAISS index
	faissIndex = &FAISSIndex{
		Vectors: make([][]float32, 0),
		Mapping: make(map[int]string),
	}

	// Load existing index if available
	loadFAISSData()
}

func hashString(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h)
}

func getEmbedding(text string) ([]float32, error) {
	ctx := context.Background()
	
	// Check Redis cache
	cacheKey := fmt.Sprintf("embedding:%s", hashString(text))
	cached, err := redisClient.Get(ctx, cacheKey).Result()
	if err == nil {
		var embedding []float32
		if err := json.Unmarshal([]byte(cached), &embedding); err == nil {
			return embedding, nil
		}
	}

	// Generate embedding using Gemini
	model := genaiClient.EmbeddingModel("text-embedding-004")
	resp, err := model.EmbedContent(ctx, genai.Text(text))
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %v", err)
	}

	embedding := make([]float32, len(resp.Embedding.Values))
	for i, v := range resp.Embedding.Values {
		embedding[i] = v
	}

	// Cache the embedding
	embeddingJSON, _ := json.Marshal(embedding)
	redisClient.Set(ctx, cacheKey, embeddingJSON, 24*time.Hour)

	return embedding, nil
}

func extractPDFText(filePath string) ([]string, error) {
	file, reader, err := pdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF: %v", err)
	}
	defer file.Close()

	totalPages := reader.NumPage()
	var chunks []string

	for i := 1; i <= totalPages; i += 5 {
		var chunk strings.Builder
		
		for j := i; j < i+5 && j <= totalPages; j++ {
			page := reader.Page(j)
			if page.V.IsNull() {
				continue
			}
			
			text, err := page.GetPlainText(nil)
			if err != nil {
				continue
			}
			
			chunk.WriteString(text)
		}
		
		if chunk.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(chunk.String()))
		}
	}

	return chunks, nil
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func searchSimilar(queryEmbedding []float32, k int) []int {
	if len(faissIndex.Vectors) == 0 {
		return []int{}
	}

	type similarity struct {
		index int
		score float32
	}

	similarities := make([]similarity, len(faissIndex.Vectors))
	for i, vector := range faissIndex.Vectors {
		similarities[i] = similarity{
			index: i,
			score: cosineSimilarity(queryEmbedding, vector),
		}
	}

	// Sort by similarity score (descending)
	for i := 0; i < len(similarities)-1; i++ {
		for j := i + 1; j < len(similarities); j++ {
			if similarities[i].score < similarities[j].score {
				similarities[i], similarities[j] = similarities[j], similarities[i]
			}
		}
	}

	// Return top k indices
	result := make([]int, 0, k)
	for i := 0; i < k && i < len(similarities); i++ {
		result = append(result, similarities[i].index)
	}

	return result
}

func saveFAISSData() error {
	indexPath := filepath.Join(kartikDir, "faiss_index.gob")
	
	file, err := os.Create(indexPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(faissIndex)
}

func loadFAISSData() error {
	indexPath := filepath.Join(kartikDir, "faiss_index.gob")
	
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		return nil // File doesn't exist, start with empty index
	}

	file, err := os.Open(indexPath)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(faissIndex)
}

func generateAnswer(contextText, question string) (string, error) {
	ctx := context.Background()
	
	prompt := fmt.Sprintf(`


Context: %s

Question: %s

Answer in 1-2 sentences. JSON format: {"answer": "brief answer"}
`, contextText[:800], question)


	model := genaiClient.GenerativeModel("gemini-2.5-pro")
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %v", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response generated")
	}

	return fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0]), nil
}

// Handlers
func uploadPDF(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file provided"})
		return
	}

	// Save uploaded file temporarily
	tempPath := fmt.Sprintf("./tmp_%s", file.Filename)
	if err := c.SaveUploadedFile(file, tempPath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
		return
	}
	defer os.Remove(tempPath)

	// Extract chunks
	chunks, err := extractPDFText(tempPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to extract PDF text"})
		return
	}

	// Generate embeddings and add to index
	for _, chunk := range chunks {
		embedding, err := getEmbedding(chunk)
		if err != nil {
			log.Printf("Failed to get embedding for chunk: %v", err)
			continue
		}

		idx := len(faissIndex.Vectors)
		faissIndex.Vectors = append(faissIndex.Vectors, embedding)
		faissIndex.Mapping[idx] = chunk
	}

	// Persist data
	if err := saveFAISSData(); err != nil {
		log.Printf("Failed to save FAISS data: %v", err)
	}

	c.JSON(http.StatusOK, gin.H{"message": "File processed and embeddings stored"})
}

func askQuestion(c *gin.Context) {
	var req AskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	if len(faissIndex.Vectors) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "FAISS index is empty"})
		return
	}

	// Get question embedding
	queryEmbedding, err := getEmbedding(req.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get question embedding"})
		return
	}

	// Search for similar chunks
	k := 3
	if len(faissIndex.Vectors) < k {
		k = len(faissIndex.Vectors)
	}
	
	similarIndices := searchSimilar(queryEmbedding, k)
	
	// Build context
	var contextParts []string
	for _, idx := range similarIndices {
		if text, exists := faissIndex.Mapping[idx]; exists {
			contextParts = append(contextParts, text)
		}
	}
	context := strings.Join(contextParts, "\n\n")

	// Generate answer
	answer, err := generateAnswer(context, req.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate answer"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"answer": answer})
}

func hackRxRun(c *gin.Context) {
	// Check authorization
	auth := c.GetHeader("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Missing or invalid Authorization header"})
		return
	}

	var req HackRxRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	if req.Documents == "" || len(req.Questions) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Missing documents URL or questions"})
		return
	}

	// Download PDF
	resp, err := http.Get(req.Documents)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to download PDF"})
		return
	}
	defer resp.Body.Close()

	// Save to temporary file
	tempFile, err := os.CreateTemp("", "hackrx_*.pdf")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create temporary file"})
		return
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	_, err = io.Copy(tempFile, resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save PDF"})
		return
	}

	// Extract chunks
	chunks, err := extractPDFText(tempFile.Name())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to extract PDF text"})
		return
	}

	// Reset index for this session
	tempIndex := &FAISSIndex{
		Vectors: make([][]float32, 0),
		Mapping: make(map[int]string),
	}

	// Generate embeddings
	for _, chunk := range chunks {
		embedding, err := getEmbedding(chunk)
		if err != nil {
			log.Printf("Failed to get embedding for chunk: %v", err)
			continue
		}

		idx := len(tempIndex.Vectors)
		tempIndex.Vectors = append(tempIndex.Vectors, embedding)
		tempIndex.Mapping[idx] = chunk
	}

	// Process questions
	var answers []string
	for _, question := range req.Questions {
		queryEmbedding, err := getEmbedding(question)
		if err != nil {
			answers = append(answers, `{"answer": "Failed to process question", "source_clause": "", "reasoning": "Error generating embedding"}`)
			continue
		}

		// Search in temporary index
		k := 3
		if len(tempIndex.Vectors) < k {
			k = len(tempIndex.Vectors)
		}

		var contextParts []string
		if k > 0 {
			similarities := make([]struct {
				index int
				score float32
			}, len(tempIndex.Vectors))

			for i, vector := range tempIndex.Vectors {
				similarities[i] = struct {
					index int
					score float32
				}{
					index: i,
					score: cosineSimilarity(queryEmbedding, vector),
				}
			}

			// Sort by similarity
			for i := 0; i < len(similarities)-1; i++ {
				for j := i + 1; j < len(similarities); j++ {
					if similarities[i].score < similarities[j].score {
						similarities[i], similarities[j] = similarities[j], similarities[i]
					}
				}
			}

			// Get top k
			for i := 0; i < k && i < len(similarities); i++ {
				if text, exists := tempIndex.Mapping[similarities[i].index]; exists {
					contextParts = append(contextParts, text)
				}
			}
		}

		context := strings.Join(contextParts, "\n\n")
		answer, err := generateAnswer(context, question)
		if err != nil {
			answers = append(answers, `{"answer": "Failed to generate answer", "source_clause": "", "reasoning": "Error in answer generation"}`)
		} else {
			answers = append(answers, answer)
		}
	}

	c.JSON(http.StatusOK, gin.H{"answers": answers})
}

func main() {
	r := gin.Default()

	// Routes
	r.POST("/upload", uploadPDF)
	r.POST("/ask", askQuestion)
	r.POST("/hackrx/run", hackRxRun)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on port %s", port)
	log.Fatal(r.Run(":" + port))
}