# AlphaEngine – Securities Signal Intelligence Platform

**AlphaEngine** is a production-ready real-time signal intelligence platform for traditional securities markets. It provides actionable trading indicators by analyzing market behavior, price movements, and (optionally) social sentiment across major ETFs and futures contracts.

Migrated from alphaengine (securities-focused), AlphaEngine uses the same battle-tested architecture but targets traditional securities with Yahoo Finance integration.

---

## 🚀 Overview

AlphaEngine ingests market data from Yahoo Finance, processes it through ML models, and produces **signal triggers** with rich provenance. Every inference, label, retrain, and threshold decision is logged, versioned, and surfaced in CI.

**Supported Tickers:**
- **SPY** – S&P 500 ETF
- **QQQ** – Nasdaq-100 ETF
- **XLF** – Financial Sector ETF
- **XLK** – Technology Sector ETF
- **ES** – E-mini S&P 500 Futures (ES=F)

---

## ✨ Key Features

- **Yahoo Finance Integration** – Real-time price data for major securities
- **FastAPI server** – Lightweight async API for scoring and feedback
- **ML Signal Generation** – Statistical + ML ensemble for trigger detection
- **Feedback-driven Learning** – Continuous model improvement loop
- **Per-origin Analytics** – Precision, recall, F1 tracking
- **Social Sentiment (Optional)** – Twitter, Reddit, and news sentiment (disabled by default)
- **Threshold Optimization** – Automated backtesting and recommendations
- **CI/CD Pipeline** – Full test coverage with GitHub Actions

---

## 🛠 Tech Stack

- **Python 3.10+**
- **FastAPI + Uvicorn** – Async REST API
- **Yahoo Finance API** – Market data source
- **scikit-learn** – ML models (logistic regression, RF, gradient boosting)
- **Redis** – Signal caching
- **pandas** – Data processing
- **VaderSentiment** – Sentiment analysis (when enabled)
- **pytest** – Test suite

---

## 📊 Data Sources

### Primary: Yahoo Finance
- Real-time price data via Yahoo Finance API v8
- No API key required
- Supports ETFs and futures contracts
- 5-minute cache TTL to respect rate limits

### Optional: Social Sentiment (Disabled by Default)
AlphaEngine includes social sentiment infrastructure but it's **disabled by default**:

- **News Sentiment** – Set `NEWS_SENTIMENT_ENABLED=true` to enable
- **Reddit Sentiment** – Set `REDDIT_SENTIMENT_ENABLED=true` to enable
- **Twitter Sentiment** – Set `TWITTER_SENTIMENT_ENABLED=true` to enable

When disabled, sentiment returns neutral (0.0) values.

---

## 🚦 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file:
```bash
AE_ENV=dev
REDIS_URL=redis://localhost:6379
CORS_ORIGINS=http://localhost:3000

# Optional: Enable social sentiment (disabled by default)
NEWS_SENTIMENT_ENABLED=false
REDDIT_SENTIMENT_ENABLED=false
TWITTER_SENTIMENT_ENABLED=false
```

### 3. Run Server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Health Check
```bash
curl http://localhost:8000/health
```

---

## 🌐 API Endpoints

| Route | Purpose |
|-------|---------|
| `/ping` | Lightweight heartbeat for uptime monitoring |
| `/health` | Comprehensive system health check |
| `/internal/trigger-likelihood/score` | Run inference on market signals |
| `/internal/trigger-likelihood/feedback` | Submit label feedback for learning |

---

## 📂 Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI entrypoint, initializes AlphaEngine backend |
| `src/price_fetcher.py` | Yahoo Finance integration for real-time prices |
| `src/mock_loader.py` | Loads mock data for supported securities |
| `src/fake_ingest.py` | Simulates realistic market data ingestion |
| `src/sentiment_*.py` | Social sentiment modules (disabled by default) |
| `src/ml/infer.py` | ML inference engine with ensemble models |
| `src/paths.py` | Path/env configuration management |

---

## 🔧 Configuration

### Environment Variables

- `AE_ENV` – Environment (dev/staging/prod)
- `REDIS_URL` – Redis connection string
- `CORS_ORIGINS` – Allowed CORS origins (comma-separated)
- `NEWS_SENTIMENT_ENABLED` – Enable news sentiment (default: false)
- `REDDIT_SENTIMENT_ENABLED` – Enable Reddit sentiment (default: false)
- `TWITTER_SENTIMENT_ENABLED` – Enable Twitter sentiment (default: false)

### Supported Tickers

Edit `src/price_fetcher.py` to add more tickers:
```python
SUPPORTED_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "XLF": "XLF",
    "XLK": "XLK",
    "ES": "ES=F",
    # Add more here
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest -q

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_price_fetcher.py -v
```

---

## 📈 Extending AlphaEngine

### Adding New Securities

1. Add ticker to `SUPPORTED_TICKERS` in `src/price_fetcher.py`
2. Update mock data in `src/fake_ingest.py` with realistic price ranges
3. Add to `ASSET_KEYWORDS` in sentiment files (if enabling sentiment)
4. Update tests to include new ticker

### Enabling Social Sentiment

To enable sentiment analysis for securities:

1. Set environment variables to `true`:
   ```bash
   NEWS_SENTIMENT_ENABLED=true
   REDDIT_SENTIMENT_ENABLED=true
   TWITTER_SENTIMENT_ENABLED=true
   ```

2. Install optional dependencies (if needed):
   ```bash
   pip install tweepy snscrape
   ```

3. Configure API keys:
   ```bash
   TWITTER_BEARER_TOKEN=your_token_here
   CRYPTOPANIC_API_KEY=your_key_here  # Or replace with securities news API
   ```

---

## 🏗 Architecture

AlphaEngine follows a modular architecture:

- **Data Layer** – Yahoo Finance API integration with caching
- **Signal Generation** – Statistical + ML models for trigger detection
- **Feedback Loop** – Continuous learning from labeled signals
- **API Layer** – RESTful endpoints for integration
- **Analytics** – Per-ticker and per-model performance tracking

---

## 📊 CI/CD Pipeline

Every commit triggers:
- Full test suite execution
- Model training and validation
- Performance metrics generation
- Artifact publishing (charts, models, reports)

View results in GitHub Actions artifacts.

---

## 🔒 Production Deployment

### Docker
```bash
docker build -t alphaengine .
docker run -p 8000:8000 --env-file .env alphaengine
```

### Environment Setup
- Set `AE_ENV=production`
- Configure Redis for persistent caching
- Set up monitoring for `/health` endpoint
- Enable HTTPS with reverse proxy (nginx/Caddy)

---

## 📝 License

See LICENSE.TXT for details.

---

## 🙏 Acknowledgments

AlphaEngine is built on the alphaengine signal engine architecture, adapted for traditional securities markets.

---

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Check documentation in `/docs`
- Review test files in `/tests` for usage examples

---

**Built for traders who demand production-grade signal intelligence.**
