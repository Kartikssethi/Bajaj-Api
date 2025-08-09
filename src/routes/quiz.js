const express = require("express");
const router = express.Router();
const QuizService = require("../services/QuizService");

const quizService = new QuizService();

// Test endpoint to solve the quiz
router.get("/solve", async (req, res) => {
  try {
    console.log("Quiz solve endpoint called");
    const flightNumber = await quizService.solveQuiz();

    res.json({
      success: true,
      flightNumber: flightNumber,
      message: "Quiz solved successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error in quiz solve endpoint:", error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

// Endpoint to get favorite city
router.get("/favorite-city", async (req, res) => {
  try {
    const favoriteCity = await quizService.getFavoriteCity();
    res.json({
      success: true,
      city: favoriteCity,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error getting favorite city:", error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

// Endpoint to test landmark mapping
router.get("/landmark/:city", (req, res) => {
  try {
    const city = req.params.city;
    const landmark = quizService.getLandmarkForCity(city);
    res.json({
      success: true,
      city: city,
      landmark: landmark,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

module.exports = router;
