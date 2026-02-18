# Cash Reconciliation – Back4App Container Deployment

## Project layout

```
.
├── app_with_mongodb.py      ← Flask app (entry point)
├── mongodb_handler.py       ← MongoDB abstraction layer
├── brokers/
│   ├── __init__.py
│   ├── clearstreet.py
│   ├── scb.py
│   ├── gtna.py
│   └── riyadhcapital.py
├── templates/
│   └── reconciliation.html
├── static/                  ← (add your JS/CSS here)
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── docker-compose.yml       ← local testing only
```

---

## 1 – Test locally first

```bash
docker compose up --build
# Open http://localhost:8080
```

---

## 2 – Push to Back4App Containers

### Prerequisites
- Back4App account with **Containers** enabled
- Back4App CLI installed: `npm install -g b4a`

### Steps

```bash
# Log in
b4a login

# Create a new container app (first time only)
b4a new

# Deploy
b4a deploy
```

Back4App will build the `Dockerfile` on their infra and expose your app on a
`*.back4app.app` subdomain.

---

## 3 – Required environment variables

Set these in **Back4App Dashboard → Containers → Your App → Environment**:

| Variable          | Example value                                    | Notes                          |
|-------------------|--------------------------------------------------|--------------------------------|
| `SECRET_KEY`      | `some-long-random-string`                        | Flask session signing key      |
| `MONGODB_URI`     | `mongodb+srv://user:pass@cluster.mongodb.net/`   | MongoDB Atlas recommended      |
| `MONGODB_DB_NAME` | `cash_recon_prod`                                | Target database name           |
| `PORT`            | `8080`                                           | Usually set automatically      |

> **Never commit real credentials** to the repository.  
> Use Back4App's environment variables UI or Secrets for sensitive values.

---

## 4 – MongoDB Atlas (recommended for production)

1. Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/cloud/atlas)
2. Add **0.0.0.0/0** to the IP Allowlist (or use Back4App's static egress IP)
3. Create a database user and copy the connection string into `MONGODB_URI`

The app falls back to file-based storage if MongoDB is unreachable, but in a
container file storage is **ephemeral** (lost on restart). Always configure
MongoDB for production.

---

## 5 – Collections used

| Collection         | Purpose                                     |
|--------------------|---------------------------------------------|
| `session_rec`      | Active & saved reconciliation DataFrames    |
| `carry_forward`    | Carried-forward unmatched rows per account  |
| `history`          | Cleared/matched breaks archive per account  |
| `accounts`         | Global account list                         |
| `session_metadata` | Misc session state                          |

---

## 6 – Health check

Back4App pings `GET /health` to verify the container is alive.  
The app already returns `200 OK` on that route.

---

## 7 – Scaling notes

- The app is **stateless by design** when MongoDB is connected (sessions are
  identified by a cookie-based UUID and data is fetched from Mongo on every
  request).
- Increase gunicorn workers in `Dockerfile` CMD for higher concurrency:
  `--workers 4 --threads 4`
- Session pickle files in `/tmp/cash_recon_pro/sessions/` act as an in-process
  cache and are automatically re-populated from MongoDB on a cache miss, so
  horizontal scaling is safe.
