# Quick Start Guide - MongoDB Integration

## Step 1: Install MongoDB
### Option A - Local Installation (Recommended for testing)
- Download MongoDB Community from: https://www.mongodb.com/try/download/community
- Install and start MongoDB service

### Option B - MongoDB Atlas (Cloud)
- Sign up at: https://www.mongodb.com/cloud/atlas
- Create a free cluster
- Get connection string

## Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Configure MongoDB (Optional)
If using custom MongoDB settings, set environment variables:

**Windows:**
```cmd
set MONGODB_URI=mongodb://localhost:27017/
set MONGODB_DB_NAME=cash_recon_db
```

**Linux/Mac:**
```bash
export MONGODB_URI=mongodb://localhost:27017/
export MONGODB_DB_NAME=cash_recon_db
```

## Step 4: Run the Application
```bash
python app_with_mongodb.py
```

## What to Expect
When the app starts, you'll see:
- `✓ MongoDB connected successfully to database: cash_recon_db` - MongoDB is working
- `✗ MongoDB connection failed: ...` - App will use file backup (still works fine)

## File Structure
```
your_project/
│
├── app_with_mongodb.py          # Main application (MongoDB-enabled)
├── mongodb_handler.py            # MongoDB integration module
├── requirements.txt              # Dependencies (includes pymongo)
├── reconciliation.html           # Frontend
│
├── brokers/                      # Broker modules directory
│   ├── __init__.py
│   ├── clearstreet.py
│   ├── scb.py
│   ├── gtna.py
│   └── riyadhcapital.py
│
└── templates/                    # (if you have templates folder)
    └── ...
```

## Need Help?
Read the full documentation in: **MONGODB_INTEGRATION_CHANGES.md**
