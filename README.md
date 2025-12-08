# Cash Reconciliation App

A Flask-based web application for cash reconciliation with MongoDB integration, ready for deployment on Railway.

## 🌟 Features

- **Multi-Broker Support**: Clear Street, SCB, GTNA, Riyadh Capital
- **MongoDB Integration**: Cloud-based data storage with MongoDB Atlas
- **Automatic Matching**: Intelligent reconciliation algorithm
- **Export to Excel**: Professional formatted reports
- **Carry Forward**: Unmatched items carry to next period
- **History Tracking**: Maintain audit trail of matched items

## 🚀 Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### One-Click Deployment

1. Click the button above
2. Connect your GitHub account
3. Deploy!

Your app will be live at: `https://your-app-name.up.railway.app`

## 📋 Manual Setup

### Prerequisites

- Python 3.11+
- MongoDB Atlas account (free tier available)
- Git

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cash-recon-app.git
cd cash-recon-app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app_with_mongodb.py
```

Visit http://localhost:8080

### Railway Deployment

See detailed instructions in [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md)

**Quick steps:**
```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main

# Deploy on Railway
# Go to railway.app and deploy from your GitHub repo
```

## 🔧 Configuration

### MongoDB Connection

Your MongoDB Atlas connection is pre-configured:
```python
MONGODB_URI = "mongodb+srv://admin:Admin123456@cluster0.z8yhqsg.mongodb.net/cash_recon"
MONGODB_DB_NAME = "cash_recon"
```

To use environment variables instead:
```bash
export MONGODB_URI="your-mongodb-connection-string"
export MONGODB_DB_NAME="your-database-name"
```

## 📁 Project Structure

```
cash-recon-app/
├── app_with_mongodb.py          # Main Flask application
├── mongodb_handler.py            # MongoDB operations
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway process definition
├── railway.json                  # Railway configuration
├── brokers/                      # Broker-specific modules
│   ├── clearstreet.py
│   ├── scb.py
│   ├── gtna.py
│   └── riyadhcapital.py
└── templates/                    # HTML templates
    └── reconciliation.html
```

## 🎯 Usage

1. **Upload Files**: Upload AT and Broker files
2. **Select Date Range**: Choose reconciliation period
3. **Auto-Match**: System automatically matches transactions
4. **Review**: Check matched and unmatched items
5. **Export**: Download Excel report

## 📊 MongoDB Collections

- `session_rec` - Active reconciliation data
- `carry_forward` - Unmatched items to carry forward
- `history` - Historical matched breaks
- `accounts` - Account list

## 🐛 Troubleshooting

### MongoDB Connection Issues

1. Check MongoDB Atlas cluster is running
2. Verify network access allows Railway IPs
3. Check connection string in code or environment variables

### Deployment Issues

Check Railway logs:
```bash
railway logs
```

Or view in Railway dashboard under "Deployments"

## 📚 Documentation

- [Railway Deployment Guide](RAILWAY_DEPLOYMENT_GUIDE.md)
- [MongoDB Integration Changes](MONGODB_INTEGRATION_CHANGES.md)
- [Quick Start Guide](QUICK_START.md)

## 🔒 Security

- MongoDB connection uses SSL/TLS
- Railway provides HTTPS by default
- Session data encrypted
- File-based fallback for redundancy

## 💰 Cost

### Free Tier (Perfect for development)
- **Railway**: $5/month free credit, 500 hours
- **MongoDB Atlas**: 512MB storage, free forever
- **Total**: Free for small-scale usage

### Paid (For production)
- **Railway**: ~$5-10/month
- **MongoDB Atlas**: $9/month (M2 cluster)
- **Total**: ~$14-19/month

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the deployment guide
- Contact support

## 🎉 Acknowledgments

- Flask framework
- MongoDB Atlas
- Railway hosting platform
- All broker API providers

---

**Made with ❤️ for cash reconciliation**
