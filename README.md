# 🏏 SportsFan360 - AI Analytics Backend

Welcome to the AI Analytics Backend for **SportsFan360**. This is a high-performance cricket analytics service powered by **FastAPI**, a robust **Pandas** datalayer loading ball-by-ball datasets, and **Groq (Llama 3.3 70B)** for natural language QA, matchups, trivia, and player statistics.

---

## 🚀 Live Deployment

The backend is fully deployed and active in the cloud on Render:
* **Base URL**: `https://sportsfan360-askai.onrender.com`
* **Health Check**: `https://sportsfan360-askai.onrender.com/health`

---

## 🛠️ Key Features

- 🧠 **Smart Cricket QA**: Natural language question answering backed by a local ball-by-ball database (using Groq).
- 🎯 **Matchup Analysis**: Advanced head-to-head metrics for batter vs bowler queries.
- 📈 **Leaderboards**: Orange Cap, Purple Cap, career averages, and season stats dynamically filtered.
- 📰 **SportsFan Feed**: Integrated news aggregator yielding trending news cards.
- ⚔️ **Player Battles**: Compare two players with a calculated statistical impact factor.
- ⚡ **Daily Challenges**: Interactive fan trivia and predictions.

---

## 📡 API Endpoints

### 1. General & Health
* **`GET /`**
  * Returns home status message.
  * *Response*: `{"message": "SportsFan360 AI running"}`
* **`GET /health`**
  * Health status check.
  * *Response*: `{"status": "ok"}`

### 2. AI Queries
* **`GET /ask?question=<your-question>`**
  * Submits a question to the AI engine powered by Groq and the local Parquet datalayer.
  * *Example*: `/ask?question=Who has most IPL runs?`
  * *Response*: 
    ```json
    {
      "answer": "**V Kohli** won the runs title in **IPL** with **9144 runs**.",
      "chart_title": "Top 1 — runs",
      "chart_data": [{"player": "V Kohli", "value": 9144.0}]
    }
    ```

### 3. Statistics & Aggregations
* **`GET /standings`**
  * Fetch current IPL points table.
* **`GET /feed`**
  * Get real-time news and insight cards.
* **`GET /player-battle?p1=<player1>&p2=<player2>`**
  * Compares two players and calculates their respective cricket impact ratings.
* **`GET /daily-challenge`**
  * Generates questions and trivia options for fans' daily predictions.

---

## 💻 Local Development

### 1. Prereqs
Ensure you have Python 3.10+ installed.

### 2. Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/guptaraghav81/agent-ai.git
cd agent-ai/agent-ai

# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root of the project:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Server
```bash
uvicorn agent:app --reload --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` to access the local server.

---

## 🎨 CORS Integration
CORS is configured with `allow_origins=["*"]`, enabling instant and secure connection to any frontend framework (Next.js, React, Mobile Apps).
