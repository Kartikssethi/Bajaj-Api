const axios = require("axios");

class QuizService {
  constructor() {
    // City to landmark mapping based on the parallel world
    this.cityToLandmark = {
      Delhi: "Gateway of India",
      Mumbai: "India Gate",
      Chennai: "Charminar",
      Hyderabad: "Marina Beach",
      Ahmedabad: "Howrah Bridge",
      Mysuru: "Golconda Fort",
      Kochi: "Qutub Minar",
      Pune: "Meenakshi Temple",
      Nagpur: "Lotus Temple",
      Chandigarh: "Mysore Palace",
      Kerala: "Rock Garden",
      Bhopal: "Victoria Memorial",
      Varanasi: "Vidhana Soudha",
      Jaisalmer: "Sun Temple",
    };

    // International cities mapping
    this.internationalCityToLandmark = {
      "New York": "Eiffel Tower",
      London: "Statue of Liberty",
      Tokyo: "Big Ben",
      Beijing: "Colosseum",
      Bangkok: "Christ the Redeemer",
      Toronto: "Burj Khalifa",
      Dubai: "CN Tower",
      Amsterdam: "Petronas Towers",
      Cairo: "Leaning Tower of Pisa",
      "San Francisco": "Mount Fuji",
      Berlin: "Niagara Falls",
      Barcelona: "Louvre Museum",
      Moscow: "Stonehenge",
      Seoul: "Sagrada Familia",
      "Cape Town": "Acropolis",
      Istanbul: "Big Ben",
      Riyadh: "Machu Picchu",
      Paris: "Taj Mahal",
      "Dubai Airport": "Moai Statues",
      Singapore: "Christchurch Cathedral",
      Jakarta: "The Shard",
      Vienna: "Blue Mosque",
      Kathmandu: "Neuschwanstein Castle",
      "Los Angeles": "Buckingham Palace",
      Mumbai: "Space Needle",
      Seoul: "Times Square",
    };

    // Landmark to flight endpoint mapping
    this.landmarkToFlightEndpoint = {
      "Gateway of India": "getFirstCityFlightNumber",
      "Taj Mahal": "getSecondCityFlightNumber",
      "Eiffel Tower": "getThirdCityFlightNumber",
      "Big Ben": "getFourthCityFlightNumber",
    };
  }

  async solveQuiz() {
    try {
      console.log("Starting quiz solving process...");

      // Step 1: Get the favorite city
      const favoriteCity = await this.getFavoriteCity();
      console.log(`Favorite city received: ${favoriteCity}`);

      // Step 2: Find the landmark for this city
      const landmark = this.getLandmarkForCity(favoriteCity);
      console.log(`Landmark for ${favoriteCity}: ${landmark}`);

      // Step 3: Get the flight number based on the landmark
      const flightNumber = await this.getFlightNumber(landmark);
      console.log(`Flight number received: ${flightNumber}`);

      return flightNumber;
    } catch (error) {
      console.error("Error solving quiz:", error);
      throw error;
    }
  }

  async getFavoriteCity() {
    try {
      const response = await axios.get(
        "https://register.hackrx.in/submissions/myFavouriteCity",
        {
          timeout: 10000,
        }
      );

      if (response.data && typeof response.data === "string") {
        return response.data.trim();
      } else if (
        response.data &&
        response.data.data &&
        response.data.data.city
      ) {
        return response.data.data.city.trim();
      } else if (response.data && response.data.city) {
        return response.data.city.trim();
      } else {
        throw new Error(
          "Unexpected response format from favorite city endpoint"
        );
      }
    } catch (error) {
      console.error("Error getting favorite city:", error.message);
      throw new Error(`Failed to get favorite city: ${error.message}`);
    }
  }

  getLandmarkForCity(city) {
    // Check Indian cities first
    if (this.cityToLandmark[city]) {
      return this.cityToLandmark[city];
    }

    // Check international cities
    if (this.internationalCityToLandmark[city]) {
      return this.internationalCityToLandmark[city];
    }

    throw new Error(`City '${city}' not found in the parallel world mapping`);
  }

  async getFlightNumber(landmark) {
    try {
      let endpoint;

      // Check if landmark has a specific endpoint
      if (this.landmarkToFlightEndpoint[landmark]) {
        endpoint = this.landmarkToFlightEndpoint[landmark];
      } else {
        // Default to fifth city flight number for all other landmarks
        endpoint = "getFifthCityFlightNumber";
      }

      const url = `https://register.hackrx.in/teams/public/flights/${endpoint}`;
      console.log(`Calling flight endpoint: ${url}`);

      const response = await axios.get(url, {
        timeout: 10000,
      });

      if (response.data && typeof response.data === "string") {
        return response.data.trim();
      } else if (
        response.data &&
        response.data.data &&
        response.data.data.flightNumber
      ) {
        return response.data.data.flightNumber.trim();
      } else if (response.data && response.data.flightNumber) {
        return response.data.flightNumber.trim();
      } else if (response.data && response.data.answer) {
        return response.data.answer.trim();
      } else {
        throw new Error("Unexpected response format from flight endpoint");
      }
    } catch (error) {
      console.error("Error getting flight number:", error.message);
      throw new Error(
        `Failed to get flight number for landmark '${landmark}': ${error.message}`
      );
    }
  }
}

module.exports = QuizService;
