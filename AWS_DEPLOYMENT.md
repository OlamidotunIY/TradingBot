# AWS EC2 Deployment Guide for ML Trading Bot

## Step 1: Create AWS Account
1. Go to https://aws.amazon.com
2. Click "Create an AWS Account"
3. Add credit card (won't be charged for free tier)

## Step 2: Launch Windows EC2 Instance

1. Go to **EC2 Dashboard** → **Launch Instance**
2. Configure:
   - **Name**: `ML-Trading-Bot`
   - **AMI**: `Microsoft Windows Server 2022 Base` (Free tier eligible)
   - **Instance type**: `t2.micro` (Free tier)
   - **Key pair**: Create new → Download `.pem` file
   - **Network**: Allow RDP (port 3389)
3. Click **Launch Instance**

## Step 3: Connect to Your Instance

1. Wait 5 minutes for Windows to boot
2. Go to **EC2 → Instances** → Select your instance
3. Click **Connect** → **RDP Client**
4. Download Remote Desktop file
5. Click **Get Password** → Upload your `.pem` file → Decrypt
6. Open the `.rdp` file and login with decrypted password

## Step 4: Install Software on EC2

Open PowerShell as Admin and run:

```powershell
# Install Python
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe" -OutFile python.exe
.\python.exe /quiet InstallAllUsers=1 PrependPath=1

# Install Git
Invoke-WebRequest -Uri "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe" -OutFile git.exe
.\git.exe /SILENT

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")
```

## Step 5: Download MetaTrader 5

1. Download from: https://www.metatrader5.com/en/download
2. Install MT5
3. Login to your OctaFX demo account

## Step 6: Upload Your Trading Bot

Option A - Git:
```powershell
git clone https://github.com/YOUR_USERNAME/TradingBot.git
cd TradingBot
pip install -r requirements.txt
```

Option B - Manual:
- Zip your TradingBot folder
- Upload via RDP drag-and-drop

## Step 7: Test the Bot

```powershell
cd C:\TradingBot
python paper_trade.py
```

## Step 8: Set Up Persistence (Task Scheduler)

For the GitHub Action to restart the bot, you must create a task named exactly `TradingBot`.

1. Open **Task Scheduler** → **Create Task**.
2. **General Tab**:
   - Name: `TradingBot`
   - ✅ Run whether user is logged on or not
   - ✅ Run with highest privileges
3. **Triggers Tab**:
   - Click **New** → At startup.
4. **Actions Tab**:
   - Click **New** → Start a program.
   - Program/script: `C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe` (Run `where python` in CMD to find your path)
   - Add arguments: `C:\TradingBot\paper_trade.py --live`
   - Start in: `C:\TradingBot`
5. **Settings Tab**:
   - ✅ Allow task to be run on demand
   - ✅ If the task is already running, the following rule applies: **Stop the existing instance**

## Step 9: Keep MT5 Running

Create a batch file `start_trading.bat`:
```batch
@echo off
start "" "C:\Program Files\MetaTrader 5\terminal64.exe" /login:YOUR_LOGIN /password:YOUR_PASSWORD /server:OctaFX-Demo
timeout /t 30
cd C:\TradingBot
python paper_trade.py --live
```

## Cost Summary

| Service | Free Tier | After 12 months |
|---------|-----------|-----------------|
| EC2 t2.micro | 750 hrs/month FREE | ~$8/month |
| Storage (30GB) | FREE | ~$3/month |

## Important Notes

⚠️ Keep your `.pem` file safe - you need it to connect!
⚠️ Set up billing alerts to avoid surprise charges
⚠️ MT5 must be running for the bot to work
