# MT5 Trading Bot

A professional algorithmic trading bot for MetaTrader 5 with multiple strategies, backtesting, risk management, and REST API.

## Features

- 🤖 **Algorithmic Trading**: Automated trading with customizable strategies
- 📊 **Multiple Strategies**: SMA Crossover, RSI, and extensible strategy framework
- 📈 **Backtesting**: Test strategies on historical data with comprehensive metrics
- ⚠️ **Risk Management**: Position sizing, drawdown control, and trade validation
- 🗄️ **Trade History**: SQLite database for storing trade records
- 🌐 **REST API**: FastAPI-powered endpoints for external integrations
- 💻 **CLI Interface**: Command-line tools for bot management

## Project Structure

```
TradingBot/
├── config/                 # Configuration files
│   ├── config.yaml        # Main settings
│   ├── strategies.yaml    # Strategy parameters
│   └── logging.yaml       # Logging configuration
├── src/
│   ├── core/              # Trading engine
│   ├── strategies/        # Trading strategies
│   ├── risk/              # Risk management
│   ├── backtest/          # Backtesting engine
│   ├── data/              # Database & models
│   └── utils/             # Utilities
├── api/                   # REST API endpoints
├── main.py                # Bot entry point
├── cli.py                 # CLI interface
└── requirements.txt       # Dependencies
```

## Installation

1. **Clone and setup virtual environment**:
   ```bash
   cd TradingBot
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure MT5 credentials**:

   Set environment variables or edit `config/config.yaml`:
   ```bash
   set MT5_LOGIN=your_login
   set MT5_PASSWORD=your_password
   set MT5_SERVER=your_server
   ```

## Usage

### Start Trading Bot
```bash
python cli.py run
```

### Start API Server
```bash
python cli.py api --port 8000
```

### Run Backtest
```bash
python cli.py backtest --strategy sma_crossover --symbol EURUSD --bars 1000
```

### Check Status
```bash
python cli.py status
```

### List Strategies
```bash
python cli.py strategies
```

## API Endpoints

Once the API server is running, visit `http://localhost:8000/docs` for interactive documentation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trade` | POST | Execute a trade |
| `/api/positions` | GET | Get open positions |
| `/api/position/{id}` | DELETE | Close position |
| `/api/strategies` | GET | List strategies |
| `/api/strategies/{name}/toggle` | POST | Toggle strategy |
| `/api/backtest` | POST | Run backtest |
| `/api/account` | GET | Account info |
| `/api/stats` | GET | Trading statistics |
| `/api/history` | GET | Trade history |

## Creating Custom Strategies

1. Create a new file in `src/strategies/`:

```python
from .base_strategy import BaseStrategy, Signal, SignalType

class MyStrategy(BaseStrategy):
    def analyze(self, data, symbol):
        # Calculate indicators
        return {'indicator': value}

    def generate_signals(self, analysis, symbol):
        # Generate trading signals
        if condition:
            return Signal(SignalType.BUY, symbol, reason="...")
        return Signal(SignalType.HOLD, symbol)
```

2. Register in `strategy_manager.py`
3. Add configuration in `config/strategies.yaml`

## License

MIT License
