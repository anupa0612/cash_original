# 🚂 Railway Deployment Guide - Cash Reconciliation App

## ✅ Your MongoDB Connection is Already Configured!

Your MongoDB Atlas connection string has been integrated into the code:
```
mongodb+srv://admin:Admin123456@cluster0.z8yhqsg.mongodb.net/cash_recon
```

---

## 📋 Pre-Deployment Checklist

✅ MongoDB Atlas connection string configured  
✅ Code updated for Railway hosting (PORT and HOST)  
✅ Browser auto-open disabled in production  
✅ All deployment files created  
✅ Ready to deploy!  

---

## 🚀 Deployment Steps

### Method 1: Deploy via GitHub (Recommended)

#### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `cash-recon-app`)
3. **Don't** initialize with README (we'll push existing code)

#### Step 2: Push Code to GitHub

```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with Railway deployment config"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

#### Step 3: Deploy on Railway

1. Go to https://railway.app/
2. Click **"Start a New Project"**
3. Click **"Deploy from GitHub repo"**
4. Authenticate with GitHub if needed
5. Select your repository: `cash-recon-app`
6. Railway will automatically:
   - Detect your Python app
   - Install dependencies from `requirements.txt`
   - Start the app with the Procfile command
7. Wait for deployment (2-5 minutes)
8. Click **"Generate Domain"** to get your public URL

**Your app will be live at:** `https://your-app-name.up.railway.app`

---

### Method 2: Deploy via Railway CLI (Alternative)

#### Step 1: Install Railway CLI

```bash
# Using npm
npm install -g @railway/cli

# Or using curl (Linux/Mac)
curl -fsSL https://railway.app/install.sh | sh
```

#### Step 2: Login to Railway

```bash
railway login
```

This will open your browser to authenticate.

#### Step 3: Initialize and Deploy

```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize Railway project
railway init

# Deploy
railway up
```

#### Step 4: Get Your URL

```bash
# Generate a domain
railway domain
```

---

## 🔧 Environment Variables (Optional)

Railway automatically detects your MongoDB connection from the code. If you want to override it:

1. Go to your Railway project dashboard
2. Click **"Variables"** tab
3. Add these variables:

```
MONGODB_URI=mongodb+srv://admin:Admin123456@cluster0.z8yhqsg.mongodb.net/cash_recon?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DB_NAME=cash_recon
```

**Note:** The connection string is already hardcoded in your app, so this is optional.

---

## 📁 Project Structure for Railway

Your project should look like this:

```
cash-recon-app/
│
├── app_with_mongodb.py          # Main Flask app
├── mongodb_handler.py            # MongoDB handler
├── requirements.txt              # Python dependencies
├── Procfile                      # Process definition
├── railway.json                  # Railway configuration
├── runtime.txt                   # Python version
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variables template
│
├── brokers/                      # Broker modules
│   ├── __init__.py
│   ├── clearstreet.py
│   ├── scb.py
│   ├── gtna.py
│   └── riyadhcapital.py
│
├── templates/                    # Flask templates
│   └── reconciliation.html
│
└── static/                       # Static files (CSS, JS)
```

---

## ✅ Post-Deployment Verification

After deployment, verify everything works:

### 1. Check Deployment Logs

In Railway dashboard:
- Click **"Deployments"** tab
- View latest deployment logs
- Look for: `✓ MongoDB connected successfully to database: cash_recon`

### 2. Test Your App

Visit your Railway URL: `https://your-app-name.up.railway.app`

You should see your cash reconciliation interface.

### 3. Test Database Connection

1. Upload a test file
2. Process reconciliation
3. Check MongoDB Atlas to see data being saved:
   - Go to https://cloud.mongodb.com/
   - Navigate to your cluster
   - Click **"Browse Collections"**
   - You should see collections: `session_rec`, `carry_forward`, `history`, `accounts`

---

## 🎛️ Railway Dashboard Features

### View Logs
```bash
# Via CLI
railway logs

# Or in dashboard: "Deployments" → Click deployment → "View Logs"
```

### Restart App
```bash
# Via CLI
railway restart

# Or in dashboard: Click "..." → "Restart"
```

### Scale Resources (if needed)
- Railway auto-scales based on usage
- Free tier: 500 hours/month, $5 credit
- Upgrade to Pro for unlimited hours

---

## 🔒 Security Recommendations

### 1. Secure MongoDB Password (Recommended)

Your current password `Admin123456` is in the code. To secure it:

**Option A: Use Railway Environment Variables**

1. In Railway dashboard → "Variables"
2. Add: `MONGODB_URI=mongodb+srv://...` (with secure password)
3. Update code to remove hardcoded password

**Option B: MongoDB IP Whitelist**

1. Go to MongoDB Atlas
2. Network Access → Add IP Address
3. Add Railway's IP or allow all (`0.0.0.0/0`)

### 2. Change Flask Secret Key

In `app_with_mongodb.py`, change:
```python
app.secret_key = "change-this-secret"
```

To a secure random key:
```python
app.secret_key = os.environ.get("SECRET_KEY", "your-super-secret-random-key-here")
```

Then add to Railway variables:
```
SECRET_KEY=generate_a_long_random_string_here
```

---

## 🐛 Troubleshooting

### Issue: "Application failed to start"

**Check:**
1. View deployment logs in Railway dashboard
2. Verify `requirements.txt` has all dependencies
3. Check Python version in `runtime.txt`

**Fix:**
```bash
railway logs
```

Look for error messages and fix accordingly.

### Issue: "MongoDB connection failed"

**Check:**
1. MongoDB Atlas cluster is running
2. Network access allows Railway IPs
3. Connection string is correct
4. Database user has proper permissions

**Fix:**
- In MongoDB Atlas → Network Access → Add IP: `0.0.0.0/0` (allow all)
- Or get Railway's static IPs and whitelist them

### Issue: "Module not found"

**Fix:**
Make sure all Python packages are in `requirements.txt`:
```bash
pip freeze > requirements.txt
```

Then redeploy.

### Issue: "Port already in use" (local only)

This won't happen on Railway. On local development:
```bash
# Find process using port 8080
lsof -i :8080  # Mac/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process or change PORT
export PORT=8081  # Mac/Linux
set PORT=8081  # Windows
```

---

## 📊 Monitoring Your App

### Railway Dashboard

- **Metrics:** CPU, Memory, Network usage
- **Logs:** Real-time application logs
- **Deployments:** Deployment history and status

### MongoDB Atlas

- **Metrics:** Database operations, connections, storage
- **Performance Advisor:** Query optimization suggestions
- **Alerts:** Set up email alerts for issues

---

## 💰 Cost Estimation

### Railway Pricing

**Free Tier:**
- $5 free credit per month
- 500 hours of execution time
- Perfect for development/testing

**Usage Plan (if needed):**
- $0.000463/GB-hour for memory
- $0.000231/vCPU-hour
- Typically: ~$5-10/month for small apps

**Hobby Plan:** $5/month for 500 hours + credits

### MongoDB Atlas

**Free Tier (M0):**
- 512 MB storage
- Perfect for development
- No credit card required

**Paid Tiers (if needed):**
- M2: $9/month (2GB storage)
- M5: $25/month (5GB storage)

---

## 🔄 Continuous Deployment

Railway automatically redeploys when you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "Update feature"
git push origin main
```

Railway will automatically:
1. Detect the push
2. Build your app
3. Deploy the new version
4. Zero-downtime deployment

---

## 📚 Useful Commands

```bash
# Check Railway status
railway status

# View environment variables
railway variables

# Open app in browser
railway open

# Link to existing project
railway link

# Run command in Railway environment
railway run python app_with_mongodb.py

# SSH into Railway environment (if needed)
railway shell
```

---

## 🎉 Success Checklist

After deployment, you should have:

✅ App running on Railway  
✅ MongoDB Atlas connected  
✅ Public URL accessible  
✅ Data persisting in MongoDB  
✅ Automatic deployments from GitHub  
✅ Free hosting (within limits)  

---

## 📞 Support Resources

- **Railway Docs:** https://docs.railway.app/
- **Railway Discord:** https://discord.gg/railway
- **MongoDB Docs:** https://docs.mongodb.com/
- **Flask Docs:** https://flask.palletsprojects.com/

---

## 🚀 You're All Set!

Your cash reconciliation app is ready to deploy on Railway with MongoDB Atlas. Just push to GitHub and deploy!

**Quick Deploy:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
# Then deploy on railway.app
```

**Need help?** Check the Railway logs or MongoDB Atlas monitoring dashboard.

---

**Last Updated:** December 8, 2025
