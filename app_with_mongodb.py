import io
import json
import pickle
import shutil
import re
import os
import sys
import socket
import time
import webbrowser
import multiprocessing
import threading
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import math
import numpy as np

# MongoDB integration
from mongodb_handler import MongoDBHandler


# ADD: Clear Street module (brokers/clearstreet.py)
from brokers import clearstreet as broker_clearstreet
from brokers import scb  # new
from brokers import riyadhcapital as broker_riyadhcapital
from brokers import gtna as broker_gtna


import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

# --------------------------------------------------------------------------------------
# Paths (PyInstaller-friendly + user-writable data directories)
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of this file
DATA_DIR = os.path.join(BASE_DIR, "data")               # ./data under project
os.makedirs(DATA_DIR, exist_ok=True)

def _base_path() -> str:
    """Where templates/static live (sys._MEIPASS inside onefile EXE)."""
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # PyInstaller extraction dir (read-only)
    return os.path.dirname(os.path.abspath(__file__))


TEMPLATES_DIR = os.path.join(_base_path(), "templates")
STATIC_DIR = os.path.join(_base_path(), "static")

# ── Detect container / PaaS environment ────────────────────────────────────
_IS_CONTAINER = bool(
    os.environ.get("BACK4APP_APP_ID")        # Back4App Containers
    or os.environ.get("RAILWAY_ENVIRONMENT") # Railway
    or os.environ.get("RENDER")              # Render
    or os.environ.get("DYNO")               # Heroku
    or os.environ.get("FLY_APP_NAME")       # Fly.io
    or os.environ.get("K_SERVICE")          # Cloud Run
)

# In containers use /tmp (ephemeral but always writable); locally use home dir.
if _IS_CONTAINER:
    _APP_ROOT = Path("/tmp/cash_recon_pro")
else:
    _APPDATA = os.environ.get("LOCALAPPDATA") or os.path.join(
        str(Path.home()), ".cash_recon_pro")
    _APP_ROOT = Path(_APPDATA) / "CashReconPro"

APP_DIR   = _APP_ROOT
TMP_ROOT  = APP_DIR / "sessions"
DATA_ROOT = APP_DIR / "data"
for d in (APP_DIR, TMP_ROOT, DATA_ROOT):
    d.mkdir(parents=True, exist_ok=True)

ACCOUNTS_JSON = DATA_ROOT / "accounts.json"

# --------------------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

HOST = "0.0.0.0"  # Listen on all interfaces for Railway
PORT = int(os.environ.get("PORT", 8080))  # Use Railway's PORT or default to 8080
URL = f"http://{HOST}:{PORT}/"

# --------------------------------------------------------------------------------------
# MongoDB Configuration
# --------------------------------------------------------------------------------------
# Initialize MongoDB handler
# MongoDB Atlas connection string
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb://test_anp:password@172.20.224.99:27017/?authSource=test_anp"
)
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "test_anp")

mongo_handler = MongoDBHandler(MONGODB_URI, MONGODB_DB_NAME)

def make_rec_key(account: str, broker: str) -> str:
    """
    Build a unique key for (Account + Broker)
    so each combo has its own reconciliation.
    """
    account = (account or "").strip()
    broker = (broker or "").strip()

    if not account:
        return "default"

    if not broker:
        return account  # if broker is empty, just use account

    return f"{account}__{broker}"

# Column constants
COL_DATE = "Date"
COL_DESC = "Description"
COL_AT = "AT"
COL_BRK = "Broker"
COL_SYMBOL = "Symbol"


def _pick_broker_key(name: str) -> str:
    """Return our canonical broker key from any label/alias."""
    k = (name or "").strip().lower()
    alias = {
        "velocity": "velocity",

        # use ONE canonical key for Clear Street
        "clear street": "clearstreet",
        "clearstreet": "clearstreet",

        # SCB aliases
        "scb": "scb",
        "standard chartered": "scb",
        "standard chartered bank": "scb",

        # Riyadh Capital aliases
        "riyadh capital": "riyadh capital",
        "riyadhcapital": "riyadh capital",
        "rc": "riyadh capital",

        "gtna": "gtna",
    }
    return alias.get(k, "velocity")


def clean_broker_dispatch(file_storage, start_date_str, end_date_str, broker_name, account_value):
    key = _pick_broker_key(broker_name)
    clean_fn = BROKER_REGISTRY[key]["clean"]
    # SCB needs account filter; others don't
    if key == "scb":
        return clean_fn(file_storage, start_date_str, end_date_str, account_value=account_value)
    try:
        return clean_fn(file_storage, start_date_str, end_date_str, account_value)
    except TypeError:
        return clean_fn(file_storage, start_date_str, end_date_str)


def build_rec_by_broker(at_df: pd.DataFrame, broker_df: pd.DataFrame, broker_name: str) -> pd.DataFrame:
    key = _pick_broker_key(broker_name)
    return BROKER_REGISTRY[key]["build"](at_df, broker_df)


# --------------------------------------------------------------------------------------
# Helper: JSON-safe utilities
# --------------------------------------------------------------------------------------

def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to Python-native JSON-safe records."""
    if df is None or df.empty:
        return []
    tmp = df.copy()
    for c in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[c]):
            tmp[c] = pd.to_datetime(
                tmp[c], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp = tmp.where(pd.notna(tmp), None)
    return json.loads(tmp.to_json(orient="records"))


def _json_ok(payload: dict, code: int = 200):
    return app.response_class(
        json.dumps(payload, ensure_ascii=False),
        status=code,
        mimetype="application/json",
    )


def _json_err(msg: str, code: int = 400):
    return _json_ok({"ok": False, "error": msg}, code=code)

# --------------------------------------------------------------------------------------
# Accounts & carry-forward helpers
# --------------------------------------------------------------------------------------


def _safe_account_name(acc: str) -> str:
    s = (acc or "").strip()
    if not s:
        return "Account"
    s = s.replace("/", "-").replace("\\", "-").replace("..", ".")
    return re.sub(r"[^A-Za-z0-9_.\- ]+", "_", s)


def _accounts_load() -> list[str]:
    # Try MongoDB first
    if mongo_handler.is_connected():
        accounts = mongo_handler.load_accounts_list()
        if accounts:
            return accounts
    
    # Fallback to file-based storage
    if not ACCOUNTS_JSON.exists():
        return []
    try:
        return json.loads(ACCOUNTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _accounts_save(lst: list[str]):
    # Save to MongoDB first
    if mongo_handler.is_connected():
        mongo_handler.save_accounts_list(lst)
    
    # Always save to file as backup
    ACCOUNTS_JSON.write_text(
        json.dumps(sorted(set([x for x in lst if x]),
                   key=str), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
ACCOUNTS_BY_BROKER_JSON = DATA_ROOT / "accounts_by_broker.json"

def _accounts_by_broker_load() -> dict:
    if not ACCOUNTS_BY_BROKER_JSON.exists():
        return {}
    try:
        return json.loads(ACCOUNTS_BY_BROKER_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _accounts_by_broker_save(mapping: dict):
    ACCOUNTS_BY_BROKER_JSON.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _account_dir(account: str) -> Path:
    d = DATA_ROOT / _safe_account_name(account)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _carry_path(account: str) -> Path:
    return _account_dir(account) / "carry_unmatched.pkl"


def _save_carry_for_account(account: str, rows_df: pd.DataFrame):
    df = rows_df.copy()
    
    # Save to MongoDB first
    if mongo_handler.is_connected():
        mongo_handler.save_carry_forward(account, df)
    
    # Always save to file as backup
    with open(_carry_path(account), "wb") as f:
        pickle.dump(df, f)


def _load_carry_for_account(account: str) -> pd.DataFrame | None:
    # Try MongoDB first
    if mongo_handler.is_connected():
        df = mongo_handler.load_carry_forward(account)
        if df is not None:
            return df
    
    # Fallback to file-based storage
    p = _carry_path(account)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------------------------------------------
# Matched history (per-account)
# --------------------------------------------------------------------------------------


def _history_path(account: str) -> Path:
    return _account_dir(account) / "matched_history.pkl"


_HIST_COLS = ["Date", "Symbol", "Description", "AT",
              "Broker", "MatchID", "Comments", "SavedAt", "_RowKey"]


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_HIST_COLS)


def _history_load(account: str) -> pd.DataFrame:
    if not account:
        return _empty_history_df()
    
    # Try MongoDB first
    if mongo_handler.is_connected():
        df = mongo_handler.load_history(account)
        if df is not None and not df.empty:
            for c in _HIST_COLS:
                if c not in df.columns:
                    df[c] = "" if c not in ("AT", "Broker") else 0.0
            return df[_HIST_COLS].copy()
    
    # Fallback to file-based storage
    p = _history_path(account)
    if not p.exists():
        return _empty_history_df()
    try:
        with open(p, "rb") as f:
            df = pickle.load(f)
        for c in _HIST_COLS:
            if c not in df.columns:
                df[c] = "" if c not in ("AT", "Broker") else 0.0
        return df[_HIST_COLS].copy()
    except Exception:
        return _empty_history_df()


def _history_write(account: str, df: pd.DataFrame):
    if not account:
        return
    
    # Save to MongoDB first
    if mongo_handler.is_connected():
        mongo_handler.save_history(account, df[_HIST_COLS])
    
    # Always save to file as backup
    with open(_history_path(account), "wb") as f:
        pickle.dump(df[_HIST_COLS], f)


def _rows_to_history_format(rows_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(
        rows_df[COL_DATE], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Symbol"] = rows_df.get(COL_SYMBOL, "").astype(str)
    out["Description"] = rows_df.get(COL_DESC, "").astype(str)
    out["AT"] = pd.to_numeric(rows_df.get(
        COL_AT, 0.0), errors="coerce").fillna(0.0).round(2)
    out["Broker"] = pd.to_numeric(rows_df.get(
        COL_BRK, 0.0), errors="coerce").fillna(0.0).round(2)
    out["MatchID"] = rows_df.get("MatchID", "").astype(str)
    out["Comments"] = rows_df.get("Comments", "").fillna("").astype(str)
    out["SavedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["_RowKey"] = (
        out["Date"].astype(str) + " | " +
        out["Symbol"].astype(str) + " | " +
        out["Description"].astype(str) + " | " +
        out["AT"].astype(str) + " | " +
        out["Broker"].astype(str) + " | " +
        out["MatchID"].astype(str)
    )
    return out[_HIST_COLS].copy()


def _history_append(account: str, rows_df: pd.DataFrame):
    if rows_df is None or rows_df.empty:
        return
    base = _history_load(account)
    new_rows = _rows_to_history_format(rows_df)
    merged = pd.concat([base, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["_RowKey"], keep="first")
    _history_write(account, merged)


def _autosave_history_from_df(df: pd.DataFrame, account: str):
    if df is None or not account:
        return
    m = (df["OurFlag"].astype(str) == "MATCHED") | (
        df["BrkFlag"].astype(str) == "MATCHED")
    if not m.any():
        return
    _history_append(
        account,
        df.loc[m, [COL_DATE, COL_SYMBOL, COL_DESC,
                   COL_AT, COL_BRK, "Comments", "MatchID"]],
    )


def _history_delete_matchids(account: str, match_ids: set[str]):
    """Remove all rows with given MatchID(s) from the per-account matched history."""
    if not account or not match_ids:
        return
    hist = _history_load(account)
    if hist.empty:
        return
    keep_mask = ~hist["MatchID"].astype(str).isin({str(m) for m in match_ids})
    _history_write(account, hist.loc[keep_mask].copy())


# --------------------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------------------
def _ensure_sid() -> str:
    sid = session.get("sid")
    if not sid:
        sid = str(uuid4())
        session["sid"] = sid
    return sid


def _sess_dir() -> Path:
    d = TMP_ROOT / _ensure_sid()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_df(df: pd.DataFrame, name: str = "rec.pkl"):
    # Get session ID
    sid = _ensure_sid()

    # --- Save to MongoDB ---
    if name == "rec.pkl" and mongo_handler.is_connected():
        # 1) Save per-browser-session copy (as you had before)
        mongo_handler.save_session_rec(session_id=sid, df=df)

        # 2) ALSO save per Account + Broker "existing rec"
        try:
            st = _load_state()
            account = (st.get("account") or "").strip()
            broker_key = _pick_broker_key(st.get("broker") or "")

            if account and broker_key:
                rec_key = make_rec_key(account, broker_key)
                mongo_handler.save_session_rec(
                    session_id=rec_key,
                    df=df,
                    metadata={
                        "account": account,
                        "broker": broker_key,
                        "saved_at": datetime.utcnow(),
                        "source": "autosync_from_working_rec",
                    },
                )
        except Exception:
            # Don't break file saving if Mongo sync fails
            traceback.print_exc()

    # --- Always save to local session folder as backup ---
    d = _sess_dir()
    with open(d / name, "wb") as f:
        pickle.dump(df, f)


def _load_df(name: str = "rec.pkl") -> pd.DataFrame | None:
    # Try MongoDB first (only for rec.pkl)
    if name == "rec.pkl" and mongo_handler.is_connected():
        sid = session.get("sid")
        if sid:
            df = mongo_handler.load_session_rec(sid)
            if df is not None:
                return df
    
    # Fallback to file-based storage
    p = _sess_dir() / name
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_state(**kv):
    """
    Merge current state and write atomically:
    write -> fsync -> os.replace so readers never see a partial file.
    """
    sdir = _sess_dir()
    sfile = sdir / "state.pkl"
    tmp = sdir / "state.pkl.tmp"

    cur = {}
    if sfile.exists() and sfile.stat().st_size > 0:
        try:
            with open(sfile, "rb") as f:
                cur = pickle.load(f) or {}
            if not isinstance(cur, dict):
                cur = {}
        except Exception:
            cur = {}

    cur.update(kv)

    with open(tmp, "wb") as f:
        pickle.dump(cur, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, sfile)


def _load_state() -> dict:
    """Best-effort load. Never raises if the pickle is empty/corrupt."""
    sfile = _sess_dir() / "state.pkl"
    if not sfile.exists() or sfile.stat().st_size == 0:
        return {}
    try:
        with open(sfile, "rb") as f:
            st = pickle.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _extract_symbol_from_desc(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    s = desc.strip()
    if "Cash dividend" in s and ":" in s and "-" in s:
        try:
            start = s.index(":") + 1
            end = s.index("-", start)
            return s[start:end].strip(" -:")
        except Exception:
            pass
    if "Cash Dividend Tax on" in s and "(" in s and "of" in s:
        try:
            start = s.index("of") + 2
            end = s.index("(", start)
            return s[start:end].strip(" -:")
        except Exception:
            pass
    tokens = [t.strip(",:;()") for t in s.split()]
    for t in tokens[::-1]:
        if 1 <= len(t) <= 6 and t.isalpha() and t.isupper():
            return t
    return ""


def _fmt_money(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def round2(x: float) -> float:
    return float(pd.Series([x]).round(2).iloc[0])


def subset_sum(values, rows, target, tol=0.01, max_n=18):
    vals = values[:max_n]
    idxs = rows[:max_n]
    stack = [(0, 0.0, [])]
    while stack:
        start, running, chosen = stack.pop()
        for i in range(start, len(vals)):
            s = running + vals[i]
            if abs(s - target) <= tol:
                return chosen + [idxs[i]]
            if s < target - tol:
                stack.append((i + 1, s, chosen + [idxs[i]]))
    return None


def rates_match(rate1: float, rate2: float) -> bool:
    r1 = float(pd.Series([rate1]).round(3).iloc[0])
    r2 = float(pd.Series([rate2]).round(3).iloc[0])
    return r1 == r2

# --------------------------------------------------------------------------------------
# Cleaning logic
# --------------------------------------------------------------------------------------


def clean_at(file_storage) -> pd.DataFrame:
    def to_num(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = (
            s.str.replace(r"[,$₹£€]", "", regex=True)
            .str.replace("\u2212", "-", regex=False)
            .str.replace(r"\s+", "", regex=True)
            .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        )
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    data = file_storage.read()
    file_storage.stream.seek(0)
    fn = (file_storage.filename or "").lower()

    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), dtype=str,
                         engine="python", sep=None)
    else:
        df = pd.read_excel(io.BytesIO(data), dtype=str)

    required = {"trans. date", "description", "debit", "credit", "balance"}
    header_idx = None
    for i, row in df.iterrows():
        vals = {str(v).strip().lower()
                for v in row.values if isinstance(v, str)}
        if required.issubset(vals):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            "Could not find AT header row with: Trans. Date, Description, Debit, Credit, Balance.")

    hdr = df.iloc[header_idx]
    colmap = {}
    for col in df.columns:
        key = str(hdr[col]).strip().lower()
        if key in required and key not in colmap:
            colmap[key] = col

    raw = df.iloc[header_idx + 1:].copy()
    raw = raw[
        [
            colmap["trans. date"],
            colmap["description"],
            colmap["debit"],
            colmap["credit"],
            colmap["balance"],
        ]
    ]
    raw.columns = ["Trans. Date", "Description", "Debit", "Credit", "Balance"]

    ob_mask = raw["Description"].astype(
        str).str.strip().str.lower() == "opening balance"
    if ob_mask.any():
        first_ob = ob_mask[ob_mask].index[0]
        raw = raw.loc[first_ob + 1:].copy()

    # AT is dd/mm/yyyy → dayfirst=True
    raw["Date"] = pd.to_datetime(
        raw["Trans. Date"], errors="coerce", dayfirst=True)
    debit = to_num(raw["Debit"]).abs()
    credit = to_num(raw["Credit"]).abs()
    amount = credit - debit  # Debit negative, Credit positive

    out = pd.DataFrame()
    out[COL_DATE] = raw["Date"]
    out[COL_DESC] = raw["Description"].astype(str)
    out[COL_AT] = amount
    out[COL_BRK] = 0.0
    out[COL_SYMBOL] = out[COL_DESC].apply(_extract_symbol_from_desc)
    out["Comments"] = ""

    out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)

    ex_mask = out[COL_DESC].str.contains(
        r"exchange\s*settlement", case=False, na=False)
    out.loc[ex_mask, COL_SYMBOL] = ""

    return out


def clean_broker(file_storage, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date_str, errors="coerce")
    end_dt = pd.to_datetime(end_date_str, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("Invalid date range. Please check From / To dates.")

    data = file_storage.read()
    file_storage.stream.seek(0)
    fn = (file_storage.filename or "").lower()

    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(io.BytesIO(data), dtype=str)

    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["settle_dt", "trade_dt",
                "trd_type", "symbol", "dispdescr", "n_amt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Broker file missing required columns: {', '.join(missing)}")

    for c in ["trd_type", "symbol", "dispdescr"]:
        df[c] = df[c].astype(str).str.strip()

    settle_dt = pd.to_datetime(
        df["settle_dt"], errors="coerce", dayfirst=False)
    report_dt = pd.to_datetime(df["trade_dt"], errors="coerce", dayfirst=False)
    amt = pd.to_numeric(df["n_amt"].str.replace(
        ",", "", regex=False), errors="coerce").fillna(0.0)

    out = pd.DataFrame(
        {
            "SettleDate": settle_dt,
            "ReportDate": report_dt,
            "Type": df["trd_type"],
            "Symbol": df["symbol"],
            "Desc": df["dispdescr"],
            "Amount": -amt,  # flip sign to align with AT
        }
    )

    nonzero = out["Amount"] != 0.0
    in_range = (out["SettleDate"] >= start_dt) & (out["SettleDate"] <= end_dt)
    out = out.loc[nonzero & in_range].copy()

    out = out.dropna(subset=["SettleDate"]).reset_index(drop=True)
    return out


def build_rec(at_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    TRADE_STOCK = {
        "Buy (Stock)", "Sell (Stock)",
        "Buy (Stock) (Cancel)", "Sell (Stock) (Cancel)",
    }
    TRADE_OPT = {
        "Buy to Open (Option)", "Buy to Close (Option)",
        "Sell to Open (Option)", "Sell to Close (Option)",
        "Buy to Open (Option) (Cancel)", "Buy to Close (Option) (Cancel)",
        "Sell to Open (Option) (Cancel)", "Sell to Close (Option) (Cancel)",
    }

    exch_rows = []

    # --- Equity EXCH by TRADE DATE ---
    stock = broker_df.loc[broker_df["Type"].isin(TRADE_STOCK)]
    if not stock.empty:
        sgrp = stock.groupby(stock["ReportDate"].dt.strftime(
            "%Y-%m-%d"))["Amount"].sum()
        for trade_date, amt in sgrp.items():
            exch_rows.append({
                COL_DATE: pd.to_datetime(trade_date),   # Trade Date
                COL_SYMBOL: "",
                COL_DESC: "Exchange Settlements - Equity",
                COL_AT: 0.0,
                COL_BRK: float(amt),
                "Comments": "",
            })

    # --- Options EXCH by TRADE DATE ---
    opt = broker_df.loc[broker_df["Type"].isin(TRADE_OPT)]
    if not opt.empty:
        ogrp = opt.groupby(opt["ReportDate"].dt.strftime(
            "%Y-%m-%d"))["Amount"].sum()
        for trade_date, amt in ogrp.items():
            exch_rows.append({
                COL_DATE: pd.to_datetime(trade_date),   # Trade Date
                COL_SYMBOL: "",
                COL_DESC: "Exchange Settlements - Options",
                COL_AT: 0.0,
                COL_BRK: float(amt),
                "Comments": "",
            })

    rec_parts = []
    if exch_rows:
        rec_parts.append(pd.DataFrame(exch_rows))

    # Keep AT rows as-is
    if not at_df.empty:
        rec_parts.append(
            at_df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]])

    # --- Non-trade broker rows should ALSO use Trade Date in the rec ---
    non_trade = broker_df.loc[~broker_df["Type"].isin(TRADE_STOCK | TRADE_OPT)]
    if not non_trade.empty:
        nt = pd.DataFrame()
        # <-- Trade Date here (key change)
        nt[COL_DATE] = non_trade["ReportDate"]
        nt[COL_SYMBOL] = non_trade["Symbol"].fillna("").astype(str)
        nt[COL_DESC] = non_trade["Desc"]
        nt[COL_AT] = 0.0
        nt[COL_BRK] = non_trade["Amount"]
        nt["Comments"] = ""
        rec_parts.append(nt)

    rec = (pd.concat(rec_parts, ignore_index=True)
           if rec_parts else pd.DataFrame(columns=[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]))

    rec = rec.sort_values([COL_DATE, COL_SYMBOL, COL_DESC]
                          ).reset_index(drop=True)
    rec["RowID"] = rec.index.astype(int) + 1
    rec["OurFlag"] = ""
    rec["BrkFlag"] = ""
    rec["MatchID"] = ""
    return rec


# ---------- Broker registry + dispatch ----------
BROKER_REGISTRY = {
    # keep all keys LOWERCASE to match _pick_broker_key
    "velocity": {
        "clean": lambda f, s, e: clean_broker(f, s, e),
        "build": lambda at, br: build_rec(at, br),
    },
    "clear street": {
        "clean": broker_clearstreet.clean_broker,
        "build": broker_clearstreet.build_rec,
    },
    "clearstreet": {
        "clean": broker_clearstreet.clean_broker,
        "build": broker_clearstreet.build_rec,
    },
    "scb": {
        "clean": scb.clean_broker,
        "build": scb.build_rec,
    },
    "riyadh capital": {
        "clean": broker_riyadhcapital.clean_broker,
        "build": broker_riyadhcapital.build_rec,
    },
    "gtna": {
        "clean": broker_gtna.clean_broker,
        "build": broker_gtna.build_rec,
    }
}

# --------------------------------------------------------------------------------------
# Auto-matching
# --------------------------------------------------------------------------------------
_rate_at_pshare = re.compile(r"@\s*([0-9]*\.?[0-9]+)\s*per\s*share", re.I)
_rate_div_per = re.compile(
    r"dividend\s*per\s*share\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
_rate_paren_per_sh = re.compile(r"\(\s*([0-9]*\.?[0-9]+)\s*/\s*sh\s*\)", re.I)
_rate_cash_div = re.compile(
    r"cash\s*dividend\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.I)
_rate_fee_paren = re.compile(r"\(\s*fee\s*@\s*([0-9]*\.?[0-9]+)\s*\)", re.I)
_rate_pct_at = re.compile(r"@\s*([0-9]*\.?[0-9]+)\s*%\s*rate", re.I)
_rate_kv_rate = re.compile(r"\bRATE\s*:\s*\$?\s*([0-9]*\.?[0-9]+)\b", re.I)


def parse_rate(desc: str) -> float:
    if not isinstance(desc, str):
        return 0.0
    m = (_rate_cash_div.search(desc) or
         _rate_div_per.search(desc) or
         _rate_paren_per_sh.search(desc) or
         _rate_kv_rate.search(desc) or  # NEW
         _rate_fee_paren.search(desc) or
         _rate_pct_at.search(desc) or
         _rate_at_pshare.search(desc))
    if m:
        return float(pd.Series([float(m.group(1))]).round(3).iloc[0])
    return 0.0


def guess_type(desc: str) -> str:
    s = (desc or "").upper()
    if ("DVP" in s) or ("RVP" in s):          # <-- add
        return "EXCHANGE"                     # <-- add
    if "EXCHANGE" in s and "SETTLEMENT" in s:
        return "EXCHANGE"
    if "ADR FEE" in s or "(FEE @" in s:
        return "ADR_FEE"
    if ("WITHHOLD" in s) or ("DIVNRA" in s) or ("DIVFT" in s) or ("CASH DIVIDEND TAX" in s) or ("NRA WITHHOLD" in s):
        return "WITHHOLDING"
    if ("CASH DIVIDEND" in s) or (" DIV:" in s) or ("/SH" in s and "DIV" in s) or ("FINAL - CASH DIVIDEND" in s):
        return "GROSS_DIV"
    if "RIGHTS DISTRIBUTION" in s:
        return "CORP_ACTION"
    return "OTHER"


def _build_key(desc: str, symbol: str) -> str:
    sdesc = (desc or "").lower()
    sym = (symbol or "").strip().upper()
    if (("exchange" in sdesc and "settlement" in sdesc) or
            ("dvp" in sdesc) or ("rvp" in sdesc)):         # <-- add
        if "option" in sdesc:
            return "EXCH_OPT"
        if "equity" in sdesc:
            return "EXCH_EQ"
        return "EXCH"
    return sym if sym else "NO_SYM"


def side_of_row(at_val: float, brk_val: float) -> str:
    a = round2(abs(float(at_val)))
    b = round2(abs(float(brk_val)))
    if a > 0 and b == 0:
        return "our"
    if b > 0 and a == 0:
        return "brk"
    if a == 0 and b == 0:
        return "empty"
    return "mixed"


def prepare_frame_for_solver(rec: pd.DataFrame) -> pd.DataFrame:
    df = rec.copy()
    df["DateKey"] = pd.to_datetime(
        df[COL_DATE], errors="coerce").dt.strftime("%Y-%m-%d")
    df["OurAbs"] = df[COL_AT].abs().round(2)
    df["BrkAbs"] = df[COL_BRK].abs().round(2)
    df["Diff"] = (df[COL_BRK] - df[COL_AT]).apply(round2)
    df["Rate"] = df[COL_DESC].apply(parse_rate)
    df["Type"] = df[COL_DESC].apply(guess_type)
    df["_Key"] = df.apply(lambda r: _build_key(
        r.get(COL_DESC, ""), r.get(COL_SYMBOL, "")), axis=1)
    df["__Side"] = df.apply(lambda r: side_of_row(
        r[COL_AT], r[COL_BRK]), axis=1)
    df["OurFlag"] = df["OurFlag"].fillna("")
    df["BrkFlag"] = df["BrkFlag"].fillna("")
    df["MatchID"] = df["MatchID"].fillna("")
    return df


def _pair_by_diff_two_row(df: pd.DataFrame, group_cols: list[str], tol: float) -> pd.DataFrame:
    out = df.copy()
    base = (out["OurFlag"] == "") & (
        out["BrkFlag"] == "") & (~out["Diff"].isna())
    if not base.any():
        return out

    match_counter = 1 + \
        out["MatchID"].astype(str).str.contains("MATCH #", na=False).sum()

    for _, g in out.loc[base].groupby(group_cols, dropna=False):
        cand = [i for i in g.index if out.at[i, "__Side"] in (
            "our", "brk") and round2(out.at[i, "Diff"]) != 0.0]
        if len(cand) < 2:
            continue
        cand.sort(key=lambda i: out.at[i, "Diff"])
        L, R = 0, len(cand) - 1
        used = set()
        while L < R:
            i, j = cand[L], cand[R]
            if i in used:
                L += 1
                continue
            if j in used:
                R -= 1
                continue
            s = round2(float(out.at[i, "Diff"] + out.at[j, "Diff"]))
            if abs(s) <= tol and out.at[i, "__Side"] != out.at[j, "__Side"]:
                tag = f"MATCH #{match_counter:05d}"
                for r in (i, j):
                    side = out.at[r, "__Side"]
                    if side == "our":
                        out.at[r, "OurFlag"] = "MATCHED"
                    elif side == "brk":
                        out.at[r, "BrkFlag"] = "MATCHED"
                    out.at[r, "MatchID"] = tag
                match_counter += 1
                used.update((i, j))
                L += 1
                R -= 1
            elif s < 0:
                L += 1
            else:
                R -= 1
    return out


def _pair_by_diff_one_to_many(df: pd.DataFrame, group_cols: list[str], tol: float, max_n: int = 18) -> pd.DataFrame:
    out = df.copy()
    base = (out["OurFlag"] == "") & (
        out["BrkFlag"] == "") & (~out["Diff"].isna())
    if not base.any():
        return out

    match_counter = 1 + \
        out["MatchID"].astype(str).str.contains("MATCH #", na=False).sum()

    for _, g in out.loc[base].groupby(group_cols, dropna=False):
        remaining = [i for i in g.index if out.at[i, "__Side"] in (
            "our", "brk") and round2(out.at[i, "Diff"]) != 0.0]
        remaining.sort(key=lambda i: abs(out.at[i, "Diff"]), reverse=True)
        used = set()
        for anchor in list(remaining):
            if anchor in used:
                continue
            if out.at[anchor, "OurFlag"] or out.at[anchor, "BrkFlag"]:
                used.add(anchor)
                continue
            a_side = out.at[anchor, "__Side"]
            a_diff = round2(float(out.at[anchor, "Diff"]))
            if a_side not in ("our", "brk") or a_diff == 0.0:
                used.add(anchor)
                continue
            target = round2(-a_diff)
            if target == 0.0:
                used.add(anchor)
                continue
            pool_idxs, pool_vals = [], []
            for j in remaining:
                if j == anchor or j in used:
                    continue
                if out.at[j, "__Side"] == a_side:
                    continue
                d = round2(float(out.at[j, "Diff"]))
                if d == 0.0:
                    continue
                if (target > 0 and d <= 0) or (target < 0 and d >= 0):
                    continue
                pool_idxs.append(j)
                pool_vals.append(abs(d))
            if not pool_idxs:
                used.add(anchor)
                continue
            order = sorted(range(len(pool_vals)), key=lambda k: -pool_vals[k])
            vals_sorted = [pool_vals[k] for k in order]
            rows_sorted = [pool_idxs[k] for k in order]
            chosen = subset_sum(vals_sorted, rows_sorted,
                                abs(target), tol, max_n=max_n)
            if not chosen:
                used.add(anchor)
                continue
            tag = f"MATCH #{match_counter:05d}"
            if a_side == "our":
                out.at[anchor, "OurFlag"] = "MATCHED"
            else:
                out.at[anchor, "BrkFlag"] = "MATCHED"
            out.at[anchor, "MatchID"] = tag
            for r in chosen:
                side = out.at[r, "__Side"]
                if side == "our":
                    out.at[r, "OurFlag"] = "MATCHED"
                else:
                    out.at[r, "BrkFlag"] = "MATCHED"
                out.at[r, "MatchID"] = tag
                used.add(r)
            used.add(anchor)
            match_counter += 1
    return out


def auto_match_no_date(df: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    out["_Rate3"] = out["Rate"].apply(
        lambda r: float(pd.Series([r]).round(3).iloc[0]))

    # existing passes ...
    if out["Type"].isin(["GROSS_DIV", "WITHHOLDING"]).any():
        out = _pair_by_diff_two_row(out, ["Symbol", "Type", "_Rate3"], tol)
        out = _pair_by_diff_one_to_many(
            out, ["Symbol", "Type", "_Rate3"], tol, max_n=18)

    if out["Type"].eq("ADR_FEE").any():
        out = _pair_by_diff_two_row(out, ["Symbol"], tol)
        out = _pair_by_diff_one_to_many(out, ["Symbol"], tol, max_n=18)

    if out["_Key"].isin(["EXCH_EQ", "EXCH_OPT", "EXCH"]).any():
        out = _pair_by_diff_two_row(out, ["_Key"], tol)
        out = _pair_by_diff_one_to_many(out, ["_Key"], tol, max_n=18)

    # NEW: DVP/RVP by date (and symbol, if present)
    if out["Type"].eq("DVP_RVP").any():
        # try symbol+date first (rare but safe), then just date
        out = _pair_by_diff_two_row(out, ["Symbol", "DateKey"], tol)
        out = _pair_by_diff_one_to_many(
            out, ["Symbol", "DateKey"], tol, max_n=18)
        out = _pair_by_diff_two_row(out, ["DateKey"], tol)
        out = _pair_by_diff_one_to_many(out, ["DateKey"], tol, max_n=18)

    # fallback for remaining gross/withholding (existing)
    still_mask = ((out["OurFlag"] == "") & (out["BrkFlag"] == "") &
                  (out["Type"].isin(["GROSS_DIV", "WITHHOLDING"])))
    if still_mask.any():
        out = _pair_by_diff_two_row(out, ["Symbol", "Type"], tol)
        out = _pair_by_diff_one_to_many(out, ["Symbol", "Type"], tol, max_n=18)

    return out

# --------------------------------------------------------------------------------------
# UI helpers (produce JSON-safe rows)
# --------------------------------------------------------------------------------------


def _df_unmatched(df: pd.DataFrame) -> list[dict]:
    # Handle no data
    if df is None or df.empty:
        return []

    out = df.copy()
    mask = (out["OurFlag"] == "") & (out["BrkFlag"] == "")
    out = out.loc[mask].copy()
    if out.empty:
        return []

    # Always numeric before math (robust in EXE)
    out[COL_AT] = pd.to_numeric(
        out.get(COL_AT, 0.0),  errors="coerce").fillna(0.0)
    out[COL_BRK] = pd.to_numeric(
        out.get(COL_BRK, 0.0), errors="coerce").fillna(0.0)

    out["Difference"] = (out[COL_BRK] - out[COL_AT]).round(2)
    out["DateKey"] = pd.to_datetime(
        out[COL_DATE], errors="coerce").dt.strftime("%Y-%m-%d")
    out[COL_SYMBOL] = out[COL_SYMBOL].fillna(
        "").astype(str).replace(["nan", "NaN"], "")
    out[COL_DESC] = out[COL_DESC].fillna("").astype(str)

    cols = ["RowID", "DateKey", "Symbol", COL_DESC,
            COL_AT, COL_BRK, "Difference", "Comments"]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].sort_values(["Symbol", "DateKey", "RowID"])
    return _to_records(out)


def _df_matched(df: pd.DataFrame) -> list[dict]:
    # Handle no data
    if df is None or df.empty:
        return []

    out = df.copy()
    mask = (out["OurFlag"] == "MATCHED") | (out["BrkFlag"] == "MATCHED")
    out = out.loc[mask].copy()
    if out.empty:
        return []

    # Always numeric before math (robust in EXE)
    out[COL_AT] = pd.to_numeric(
        out.get(COL_AT, 0.0),  errors="coerce").fillna(0.0)
    out[COL_BRK] = pd.to_numeric(
        out.get(COL_BRK, 0.0), errors="coerce").fillna(0.0)

    out["Difference"] = (out[COL_BRK] - out[COL_AT]).round(2)
    out["DateKey"] = pd.to_datetime(
        out[COL_DATE], errors="coerce").dt.strftime("%Y-%m-%d")
    out[COL_SYMBOL] = out[COL_SYMBOL].fillna(
        "").astype(str).replace(["nan", "NaN"], "")
    out[COL_DESC] = out[COL_DESC].fillna("").astype(str)

    cols = ["RowID", "MatchID", "DateKey", "Symbol",
            COL_DESC, COL_AT, COL_BRK, "Difference", "Comments"]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].sort_values(["MatchID", "Symbol", "DateKey", "RowID"])
    return _to_records(out)


# --------------------------------------------------------------------------------------
# Previous Rec import readers (no comments carry)
# --------------------------------------------------------------------------------------
def _read_prev_rec_xlsx(data: bytes) -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(data))
    if "Unmatched" in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name="Unmatched", dtype=str)
    elif "Recon" in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name="Recon", header=8, dtype=str)
    else:
        df = pd.read_excel(xl, sheet_name=0, dtype=str)

    df.columns = [str(c).strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            k = n.lower()
            if k in cmap:
                return cmap[k]
        return None

    c_date = pick("Date", "DateKey")
    c_sym = pick("Symbol")
    c_desc = pick("Description")
    c_at = pick("AT")
    c_brk = pick("Broker")
    c_com = pick("Comments")

    if any(x is None for x in [c_date, c_desc, c_at, c_brk]):
        raise ValueError(
            "Previous rec is missing required columns (Date/AT/Broker/Description).")

    def money(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace(",", "", regex=False).str.replace(
            r"\(([^)]+)\)", r"-\1", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        COL_DATE: pd.to_datetime(df[c_date], errors="coerce"),
        COL_SYMBOL: (df[c_sym] if c_sym else "").astype(str).replace(["nan", "NaN"], "").str.strip(),
        COL_DESC: df[c_desc].astype(str).replace(["nan", "NaN"], "").str.strip(),
        COL_AT: money(df[c_at]),
        COL_BRK: money(df[c_brk]),
        "Comments": (df[c_com].where(df[c_com].notna(), "") if c_com else "")
    })
    out = out.loc[(out[COL_AT] != 0.0) | (out[COL_BRK] != 0.0)
                  | (out["Comments"] != "")].copy()
    out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)
    return out


def _read_prev_rec_csv(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            k = n.lower()
            if k in cmap:
                return cmap[k]
        return None

    c_date = pick("Date", "DateKey")
    c_sym = pick("Symbol")
    c_desc = pick("Description")
    c_at = pick("AT")
    c_brk = pick("Broker")
    c_com = pick("Comments")

    if any(x is None for x in [c_date, c_desc, c_at, c_brk]):
        raise ValueError(
            "CSV missing required columns (Date/AT/Broker/Description).")

    def money(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace(",", "", regex=False).str.replace(
            r"\(([^)]+)\)", r"-\1", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        COL_DATE: pd.to_datetime(df[c_date], errors="coerce"),
        COL_SYMBOL: (df[c_sym] if c_sym else "").astype(str).replace(["nan", "NaN"], "").str.strip(),
        COL_DESC: df[c_desc].astype(str).replace(["nan", "NaN"], "").str.strip(),
        COL_AT: money(df[c_at]),
        COL_BRK: money(df[c_brk]),
        "Comments": (df[c_com].where(df[c_com].notna(), "") if c_com else "")
    })
    out = out.loc[(out[COL_AT] != 0.0) | (out[COL_BRK] != 0.0)
                  | (out["Comments"] != "")].copy()
    out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)
    return out

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@app.route("/")
def index():
    _ensure_sid()
    st = _load_state()
    accounts = _accounts_load()
    return render_template(
        "reconciliation.html",
        stage="upload",
        tol=st.get("tol", 0.01),
        accounts=accounts,
        account=st.get("account", ""),
        broker=st.get("broker", "Velocity"),  # NEW
    )


@app.route("/health")
def health():
    return "OK", 200

@app.route("/view_rec", methods=["POST"])
def view_rec():
    try:
        # We expect JSON: { "account": "...", "broker": "..." }
        data = request.get_json(force=True) or {}

        account = (data.get("account") or "").strip()
        broker_name = (data.get("broker") or "").strip()

        if not account:
            return jsonify(ok=False, error="Please select an account first."), 400

        # Normalize broker & build the same key used when saving
        broker_key = _pick_broker_key(broker_name)
        rec_key = make_rec_key(account, broker_key)

        # Only look in persistent store for this Account+Broker
        df = None
        if mongo_handler.is_connected():
            df = mongo_handler.load_session_rec(rec_key)

        # Nothing found → clear, friendly message back to UI
        if df is None or df.empty:
            return (
                jsonify(
                    ok=False,
                    error=(
                        "No existing reconciliation found for this Account + Broker. "
                        "Please build a new rec first."
                    ),
                ),
                404,
            )

        # Make sure the frame looks like a normal working rec
        for col in [
            COL_DATE,
            COL_SYMBOL,
            COL_DESC,
            COL_AT,
            COL_BRK,
            "Comments",
            "OurFlag",
            "BrkFlag",
            "MatchID",
        ]:
            if col not in df.columns:
                df[col] = "" if col not in (COL_AT, COL_BRK) else 0.0

        df = df[
            [
                COL_DATE,
                COL_SYMBOL,
                COL_DESC,
                COL_AT,
                COL_BRK,
                "Comments",
                "OurFlag",
                "BrkFlag",
                "MatchID",
            ]
        ].copy()
        df["RowID"] = range(1, len(df) + 1)

        # ✅ First update state so _save_df writes under the correct account+broker combo
        _save_state(account=account, broker=broker_key)

        # Then save as the active session rec
        _save_df(df, "rec.pkl")

        # Frontend will redirect to /recon after success
        return jsonify(ok=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=f"Error loading existing reconciliation: {e}"), 500






@app.route("/build_rec", methods=["POST"], endpoint="build_rec")
def build_rec_route():
    try:
        at_file = request.files.get("at_file")
        start_date = request.form.get("start_date", "").strip()
        end_date = request.form.get("end_date", "").strip()
        account = request.form.get("account", "").strip()
        broker_name = (request.form.get("broker", "")
                       or "Velocity").strip()  # NEW
        broker_key = _pick_broker_key(broker_name)  # normalize once

        # --- broker files: single vs multi (GTNA supports multiple CSVs) ---
        if broker_key in {"gtna", "gtn a", "gtn", "gtn asia"}:
            broker_files = [
                fs for fs in request.files.getlist("broker_file") if fs]
        else:
            bf = request.files.get("broker_file")
            broker_files = [bf] if bf else []

        # Basic validation (allow multiple files for GTNA)
        if not at_file or not start_date or not end_date or len(broker_files) == 0:
            accounts = _accounts_load()
            return render_template(
                "reconciliation.html",
                stage="upload",
                upload_error="Please provide both files and a valid date range.",
                accounts=accounts,
                account=account,
                broker=broker_name,  # NEW
            )

        # Parse our AT file (unchanged)
        at_df = clean_at(at_file)

        # NEW: Clean broker file(s) by selected broker
        account = (request.form.get("account") or "").strip()

        if broker_key in {"gtna", "gtn a", "gtn", "gtn asia"}:
            parts = []
            for fs in broker_files:
                try:
                    df_part = clean_broker_dispatch(
                        fs, start_date, end_date, broker_name, account)
                except Exception:
                    df_part = pd.DataFrame(
                        columns=["SettleDate", "ReportDate", "Symbol", "Description", "Amount"])
                if df_part is not None and not df_part.empty:
                    parts.append(df_part)
            broker_file = None  # keep old variable name unused for GTNA path
            broker_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
                columns=["SettleDate", "ReportDate",
                         "Symbol", "Description", "Amount"]
            )
        else:
            # single-file path for other brokers (unchanged behavior)
            broker_file = broker_files[0] if broker_files else None
            broker_df = clean_broker_dispatch(
                broker_file, start_date, end_date, broker_name, account)

        # NEW: Build rec according to broker rules
        rec = build_rec_by_broker(at_df, broker_df, broker_name)

        # Ensure numeric columns exist and are numeric (even when rec is empty)
        for c in (COL_AT, COL_BRK):
            if c not in rec.columns:
                rec[c] = 0.0
        rec[COL_AT] = pd.to_numeric(rec[COL_AT], errors="coerce").fillna(0.0)
        rec[COL_BRK] = pd.to_numeric(rec[COL_BRK], errors="coerce").fillna(0.0)

        # 1) Always save as active session rec for this browser tab
        _save_df(rec, "rec.pkl")

        # 2) Also save a persistent copy per Account + Broker
        rec_key = make_rec_key(account, broker_key)
        if mongo_handler.is_connected():
            mongo_handler.save_session_rec(
                session_id=rec_key,
                df=rec,
                metadata={
                    "account": account,
                    "broker": broker_key,
                    "broker_name": broker_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "saved_at": datetime.utcnow(),
                }
            )


        # keep your state logic (for tolerance, EB, etc.)
        _save_state(
            tol=0.01,
            eb_at=None,
            eb_brk=None,
            account=account,
            recon_date="",
            broker=broker_key  # persist normalized broker
        )


        um = _df_unmatched(rec)
        stats = {
            "matched_rows": 0,
            "total_rows": int(len(rec)),
            "symbols": sorted([s for s in pd.Series([r.get("Symbol") or "" for r in um]).unique() if s]),
        }
        st = _load_state()
        accounts = _accounts_load()
        return render_template(
            "reconciliation.html",
            stage="review",
            um=um,
            stats=stats,
            tol=st.get("tol", 0.01),
            eb_at=st.get("eb_at"),
            eb_brk=st.get("eb_brk"),
            account=st.get("account", ""),
            recon_date=st.get("recon_date", ""),
            accounts=accounts,
            broker=st.get("broker", "Velocity"),  # NEW
        )
    except Exception as e:
        traceback.print_exc()
        accounts = _accounts_load()
        return render_template(
            "reconciliation.html",
            stage="upload",
            upload_error=f"Build failed: {e}",
            accounts=accounts,
            account=request.form.get("account", "").strip(),
            broker=request.form.get("broker", "Velocity"),  # NEW
        )




@app.route("/run_automatch", methods=["POST"])
def run_automatch():
    try:
        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        body = request.get_json(silent=True) or {}
        st = _load_state()

        # Use tol from request if provided; otherwise fall back to saved state; default 0.01.
        tol = None
        try:
            tol = float(body.get("tol", ""))
        except Exception:
            tol = None
        if tol is None:
            try:
                tol = float(st.get("tol", 0.01))
            except Exception:
                tol = 0.01

        # IMPORTANT: do not call _save_state() here — avoid races with save_meta.
        # If you still want to persist tol on click, wrap in a best-effort try:
        # try: _save_state(tol=tol) except Exception: pass

        df2 = prepare_frame_for_solver(df)
        df2 = auto_match_no_date(df2, tol=tol)
        _save_df(df2, "rec.pkl")

        account = (st.get("account") or "").strip()
        _autosave_history_from_df(df2, account)

        um = _df_unmatched(df2)
        matched = _df_matched(df2)
        stats = {"matched_rows": len(matched), "total_rows": int(len(df2))}
        return _json_ok({"ok": True, "um": um, "matched": matched, "stats": stats})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Auto-match error: {e}", 500)


@app.route("/view_matched", methods=["GET"])
def view_matched():
    df = _load_df("rec.pkl")
    accounts = _accounts_load()
    if df is None:
        return render_template("reconciliation.html", stage="upload", accounts=accounts)
    matched = _df_matched(df)
    st = _load_state()
    return render_template(
        "reconciliation.html",
        stage="matched",
        matched=matched,
        eb_at=st.get("eb_at"),
        eb_brk=st.get("eb_brk"),
        account=st.get("account", ""),
        recon_date=st.get("recon_date", ""),
        tol=st.get("tol", 0.01),
        accounts=accounts,
        broker=st.get("broker", "Velocity"),  # ✅ keep broker dropdown in sync
    )


@app.route("/recon", methods=["GET"])
def recon():
    df = _load_df("rec.pkl")
    accounts = _accounts_load()
    if df is None:
        return render_template("reconciliation.html", stage="upload", accounts=accounts)
    um = _df_unmatched(df)
    st = _load_state()
    stats = {
        "matched_rows": len(_df_matched(df)),
        "total_rows": int(len(df)),
        "symbols": sorted([s for s in pd.Series([r.get("Symbol") or "" for r in um]).unique() if s]),
    }
    return render_template(
        "reconciliation.html",
        stage="review",
        um=um,
        stats=stats,
        tol=st.get("tol", 0.01),
        eb_at=st.get("eb_at"),
        eb_brk=st.get("eb_brk"),
        account=st.get("account", ""),
        recon_date=st.get("recon_date", ""),
        accounts=accounts,
        broker=st.get("broker", "Velocity"),  # ✅ keeps selected broker on screen
    )



@app.route("/manual_pair", methods=["POST"])
def manual_pair():
    try:
        body = request.get_json(silent=True) or {}
        rowids = body.get("rowids") or []

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        map_rowid_to_idx = dict(zip(df["RowID"].astype(int), df.index))
        try:
            idxs = [map_rowid_to_idx[int(x)] for x in rowids]
        except Exception:
            return _json_err("Some RowIDs not found", 400)

        if len(idxs) < 2:
            return _json_err("Pick at least two rows to pair", 400)

        for i in idxs:
            if (df.at[i, "OurFlag"] != "") or (df.at[i, "BrkFlag"] != ""):
                return _json_err(f"Row {int(df.at[i, 'RowID'])} already matched", 400)
            a = abs(float(df.at[i, COL_AT]))
            b = abs(float(df.at[i, COL_BRK]))
            if a > 0 and b > 0:
                return _json_err(f"Row {int(df.at[i, 'RowID'])} has both AT and Broker non-zero (mixed).", 400)

        st = _load_state()
        tol = float(st.get("tol", 0.01))
        sum_diff = round(
            float((df.loc[idxs, COL_BRK] - df.loc[idxs, COL_AT]).sum()), 2)
        if abs(sum_diff) > tol:
            return _json_err(
                f"Selection doesn't balance by Difference: Σ(Broker-AT)={sum_diff:.2f} (tol={tol:.2f})", 400
            )

        tag = f"MANUAL #{1 + df['MatchID'].astype(str).str.contains('#', na=False).sum():05d}"
        for i in idxs:
            a = abs(float(df.at[i, COL_AT]))
            b = abs(float(df.at[i, COL_BRK]))
            if a > 0 and b == 0:
                df.at[i, "OurFlag"] = "MATCHED"
            elif b > 0 and a == 0:
                df.at[i, "BrkFlag"] = "MATCHED"
            df.at[i, "MatchID"] = tag

        _save_df(df, "rec.pkl")

        account = (st.get("account") or "").strip()
        _autosave_history_from_df(df, account)

        um = _df_unmatched(df)
        stats = {"matched_rows": len(
            _df_matched(df)), "total_rows": int(len(df))}
        return _json_ok({"ok": True, "um": um, "stats": stats})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Manual pair error: {e}", 500)


@app.route("/recall_matched", methods=["POST"])
def recall_matched():
    try:
        body = request.get_json(silent=True) or {}
        rowids = body.get("rowids") or []
        if not rowids:
            return _json_err("Select one or more matched rows", 400)

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        # Map RowID -> df index
        map_rowid_to_idx = dict(zip(df["RowID"].astype(int), df.index))
        try:
            idxs = [map_rowid_to_idx[int(x)] for x in rowids]
        except Exception:
            return _json_err("Some RowIDs not found", 400)

        # Collect selected MatchIDs (non-empty only)
        selected_match_ids = {
            str(df.at[i, "MatchID"]).strip()
            for i in idxs
            if str(df.at[i, "MatchID"]).strip()
        }

        changed = 0

        if selected_match_ids:
            # GROUP RECALL: unmatch all rows that share any of these MatchIDs
            group_mask = df["MatchID"].astype(str).isin(selected_match_ids)
            for i in df.index[df.index.isin(df.index[group_mask])]:
                if df.at[i, "OurFlag"] == "MATCHED" or df.at[i, "BrkFlag"] == "MATCHED":
                    df.at[i, "OurFlag"] = ""
                    df.at[i, "BrkFlag"] = ""
                    df.at[i, "MatchID"] = ""
                    changed += 1

            # Also remove these MatchIDs from per-account history
            st = _load_state()
            account = (st.get("account") or "").strip()
            _history_delete_matchids(account, selected_match_ids)
        else:
            # FALLBACK: no MatchID on selected rows → behave like row-by-row recall
            for i in idxs:
                if df.at[i, "OurFlag"] == "MATCHED" or df.at[i, "BrkFlag"] == "MATCHED":
                    df.at[i, "OurFlag"] = ""
                    df.at[i, "BrkFlag"] = ""
                    df.at[i, "MatchID"] = ""
                    changed += 1

        _save_df(df, "rec.pkl")

        matched = _df_matched(df)
        return _json_ok({"ok": True, "changed": changed, "matched": matched})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Recall error: {e}", 500)


@app.route("/update_comment", methods=["POST"])
def update_comment():
    try:
        body = request.get_json(silent=True) or {}
        rowid = body.get("rowid")
        text = body.get("text", "")

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        try:
            idx = dict(zip(df["RowID"].astype(int), df.index))[int(rowid)]
        except Exception:
            return _json_err("RowID not found", 400)

        df.at[idx, "Comments"] = "" if pd.isna(text) else str(text)
        _save_df(df, "rec.pkl")
        return _json_ok({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Comment error: {e}", 500)

# ---------- NEW: Update Symbol ----------


@app.route("/update_symbol", methods=["POST"])
def update_symbol():
    try:
        body = request.get_json(silent=True) or {}
        rowid = body.get("rowid")
        sym = (body.get("symbol") or "").strip().upper()

        if rowid is None:
            return _json_err("RowID is required", 400)

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        try:
            idx = dict(zip(df["RowID"].astype(int), df.index))[int(rowid)]
        except Exception:
            return _json_err("RowID not found", 400)

        # Update and persist
        df.at[idx, COL_SYMBOL] = sym
        _save_df(df, "rec.pkl")
        return _json_ok({"ok": True, "rowid": int(rowid), "symbol": sym})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Symbol update error: {e}", 500)


@app.route("/save_meta", methods=["POST"])
def save_meta():
    try:
        body = request.get_json(silent=True) or {}
        st = _load_state()
        if "eb_at" in body:
            try:
                st["eb_at"] = float(body["eb_at"])
            except Exception:
                st["eb_at"] = None
        if "eb_brk" in body:
            try:
                st["eb_brk"] = float(body["eb_brk"])
            except Exception:
                st["eb_brk"] = None
        if "account" in body:
            st["account"] = (body.get("account") or "").strip()
        if "recon_date" in body:
            st["recon_date"] = (body.get("recon_date") or "").strip()
        if "tol" in body:
            try:
                st["tol"] = float(body["tol"])
            except Exception:
                pass
        # NEW: persist broker if sent
        if "broker" in body:
            st["broker"] = (body.get("broker") or "").strip()

        _save_state(**st)
        return _json_ok({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Meta save error: {e}", 500)

# ---------- Accounts API ----------


@app.route("/accounts/list", methods=["GET"])
def accounts_list():
    broker_name = (request.args.get("broker") or "").strip()
    all_accounts = _accounts_load()

    # No broker passed → old behaviour, show all accounts
    if not broker_name:
        return _json_ok({"accounts": all_accounts})

    # Normalize broker
    broker_key = _pick_broker_key(broker_name)

    # Load mapping { broker_key: [acc1, acc2, ...] }
    mapping = _accounts_by_broker_load()
    by_broker = mapping.get(broker_key, [])

    # Safety: keep only accounts that still exist globally
    filtered = [a for a in by_broker if a in all_accounts]

    return _json_ok({"accounts": filtered})



@app.route("/accounts/add", methods=["POST"])
def accounts_add():
    try:
        body = request.get_json(silent=True) or {}
        acc = (body.get("account") or "").strip()
        broker_name = (body.get("broker") or "").strip()

        if not acc:
            return _json_err("Account cannot be empty", 400)

        # Normal global list (same as before)
        accounts = _accounts_load()
        if acc not in accounts:
            accounts.append(acc)
            _accounts_save(accounts)

        # NEW: link account to broker
        if broker_name:
            broker_key = _pick_broker_key(broker_name)
            mapping = _accounts_by_broker_load()
            current = set(mapping.get(broker_key, []))
            current.add(acc)
            mapping[broker_key] = sorted(current)
            _accounts_by_broker_save(mapping)

        st = _load_state()
        st["account"] = acc
        _save_state(**st)

        return _json_ok({"ok": True, "accounts": _accounts_load(), "selected": acc})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Accounts add error: {e}", 500)



@app.route("/accounts/delete", methods=["POST"])
def accounts_delete():
    try:
        body = request.get_json(silent=True) or {}
        acc = (body.get("account") or "").strip()
        if not acc:
            return _json_err("Account cannot be empty", 400)
        accounts = _accounts_load()
        accounts = [a for a in accounts if a != acc]
        _accounts_save(accounts)
        try:
            shutil.rmtree(_account_dir(acc), ignore_errors=True)
        except Exception:
            pass
        st = _load_state()
        if st.get("account") == acc:
            st["account"] = ""
            _save_state(**st)
        return _json_ok({"ok": True, "accounts": _accounts_load(), "selected": st.get("account", "")})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Accounts delete error: {e}", 500)

# ---------- Carry-forward API ----------


@app.route("/carry/save", methods=["POST"])
def carry_save():
    try:
        st = _load_state()
        account = (st.get("account") or "").strip()
        if not account:
            return _json_err("Select an account first", 400)
        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)
        um = pd.DataFrame(_df_unmatched(df))
        if um.empty:
            return _json_err("No unmatched rows to save", 400)
        out = pd.DataFrame({
            COL_DATE: pd.to_datetime(um["DateKey"], errors="coerce"),
            COL_SYMBOL: um["Symbol"].astype(str),
            COL_DESC: um["Description"].astype(str),
            COL_AT: pd.to_numeric(um["AT"], errors="coerce").fillna(0.0),
            COL_BRK: pd.to_numeric(um["Broker"], errors="coerce").fillna(0.0),
            "Comments": um.get("Comments", "").astype(str),   # <— preserve
        })
        out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)
        _save_carry_for_account(account, out)
        return _json_ok({"ok": True, "saved": int(len(out))})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Carry save error: {e}", 500)


@app.route("/carry/load", methods=["POST"])
def carry_load():
    try:
        st = _load_state()
        account = (st.get("account") or "").strip()
        if not account:
            return _json_err("Select an account first", 400)
        prev = _load_carry_for_account(account)
        if prev is None or prev.empty:
            return _json_err("No carry-forward saved for this account", 404)
        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)
        add = prev.copy()
        # add["Comments"] = ""   # keep preserved comments
        add["OurFlag"] = ""
        add["BrkFlag"] = ""
        add["MatchID"] = ""
        base = df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK,
                   "Comments", "OurFlag", "BrkFlag", "MatchID"]].copy()
        merged = pd.concat([base, add], ignore_index=True)
        merged["RowID"] = merged.index.astype(int) + 1
        _save_df(merged, "rec.pkl")
        return _json_ok({"ok": True, "added": int(len(add)), "um": _df_unmatched(merged)})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Carry load error: {e}", 500)

# ---------- Previous Rec import ----------


@app.route("/import_previous_rec", methods=["POST"])
def import_previous_rec():
    try:
        file = request.files.get("prev_rec")
        if not file:
            return _json_err("Please choose a previous rec file (.xlsx or .csv)", 400)

        data = file.read()
        file.stream.seek(0)
        fn = (file.filename or "").lower()

        if fn.endswith(".xlsx"):
            new_rows = _read_prev_rec_xlsx(data)
        elif fn.endswith(".csv"):
            new_rows = _read_prev_rec_csv(data)
        else:
            return _json_err("Unsupported file type. Upload .xlsx or .csv", 400)

        if new_rows.empty:
            return _json_err("No usable rows were found in the file", 400)

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        # Normalize columns and flags
        new_rows = new_rows[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]].copy()
        new_rows["OurFlag"] = ""
        new_rows["BrkFlag"] = ""
        new_rows["MatchID"] = ""

        base = df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK,
                   "Comments", "OurFlag", "BrkFlag", "MatchID"]].copy()
        merged = pd.concat([base, new_rows], ignore_index=True)
        merged["RowID"] = merged.index.astype(int) + 1

        # 1) Save to current session
        _save_df(merged, "rec.pkl")

        # 2) ALSO persist to Mongo for this Account + Broker
        st = _load_state()
        account = (st.get("account") or "").strip()
        broker_key = _pick_broker_key(st.get("broker") or "")
        if account and mongo_handler.is_connected():
            rec_key = make_rec_key(account, broker_key)
            mongo_handler.save_session_rec(
                session_id=rec_key,
                df=merged,
                metadata={
                    "account": account,
                    "broker": broker_key,
                    "saved_at": datetime.utcnow(),
                    "source": "import_previous_rec",
                },
            )

        return _json_ok({"ok": True, "added": int(len(new_rows)), "um": _df_unmatched(merged)})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Import failed: {e}", 500)


# ---------- Manual Transactions Import (NEW) ----------


@app.route("/manual_add", methods=["POST"])
def manual_add():
    """
    Upload manual transactions as CSV/XLSX with headers:
    Date, Symbol, Description, amount, side

    side: 'A' -> goes to AT column
          'B' -> goes to Broker column

    Amount is used as provided (no sign flipping here).
    """
    try:
        file = request.files.get("tx_file")
        if not file:
            return _json_err("Please choose a file (.csv or .xlsx)", 400)

        data = file.read()
        file.stream.seek(0)
        fn = (file.filename or "").lower()

        # Read CSV/XLSX into string-typed dataframe for robust cleaning
        if fn.endswith(".csv"):
            df_in = pd.read_csv(io.BytesIO(data), dtype=str)
        elif fn.endswith(".xlsx"):
            df_in = pd.read_excel(io.BytesIO(data), dtype=str)
        else:
            return _json_err("Unsupported file type. Upload .csv or .xlsx", 400)

        # Normalize columns
        df_in.columns = [str(c).strip() for c in df_in.columns]
        cmap = {c.lower(): c for c in df_in.columns}
        required = ["date", "symbol", "description", "amount", "side"]
        missing = [c for c in required if c not in cmap]
        if missing:
            return _json_err(f"Missing required headers: {', '.join(missing)}", 400)

        c_date = cmap["date"]
        c_sym = cmap["symbol"]
        c_desc = cmap["description"]
        c_amt = cmap["amount"]
        c_side = cmap["side"]

        # Parse values
        def money(series: pd.Series) -> pd.Series:
            s = series.astype(str)
            # strip commas and convert parentheses negatives
            s = s.str.replace(",", "", regex=False).str.replace(
                r"\(([^)]+)\)", r"-\1", regex=True)
            return pd.to_numeric(s, errors="coerce").fillna(0.0)

        add = pd.DataFrame()
        # Try dd/mm/yyyy first, then mm/dd/yyyy, and use whichever parses
        d_dayfirst = pd.to_datetime(
            df_in[c_date], errors="coerce", dayfirst=True)
        d_monthfirst = pd.to_datetime(
            df_in[c_date], errors="coerce", dayfirst=False)
        add[COL_DATE] = d_dayfirst.fillna(d_monthfirst)
        add[COL_SYMBOL] = df_in[c_sym].astype(
            str).replace(["nan", "NaN"], "").str.strip()
        add[COL_DESC] = df_in[c_desc].astype(
            str).replace(["nan", "NaN"], "").str.strip()
        amt_series = money(df_in[c_amt])
        side_series = df_in[c_side].astype(str).str.strip().str.upper()

        # Map side -> AT/Broker columns (no sign flip)
        add[COL_AT] = 0.0
        add[COL_BRK] = 0.0
        mask_a = side_series.eq("A")
        mask_b = side_series.eq("B")
        add.loc[mask_a, COL_AT] = amt_series[mask_a]
        add.loc[mask_b, COL_BRK] = amt_series[mask_b]

        # Discard rows without a valid Date
        add = add.dropna(subset=[COL_DATE]).reset_index(drop=True)

        if add.empty:
            return _json_err("No valid rows (check Date formats & side A/B)", 400)

        # Default flags
        add["Comments"] = ""
        add["OurFlag"] = ""
        add["BrkFlag"] = ""
        add["MatchID"] = ""

        # Merge into current rec
        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data. Build a reconciliation first.", 400)

        base = df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK,
                   "Comments", "OurFlag", "BrkFlag", "MatchID"]].copy()
        merged = pd.concat([base, add], ignore_index=True)
        merged["RowID"] = merged.index.astype(int) + 1
        _save_df(merged, "rec.pkl")

        return _json_ok({"ok": True, "added": int(len(add)), "um": _df_unmatched(merged)})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"Manual add failed: {e}", 500)

# ---------- History API ----------


@app.route("/history/list", methods=["GET"])
def history_list():
    try:
        st = _load_state()
        account = (st.get("account") or "").strip()
        hist = _history_load(account)
        cols = ["Date", "Symbol", "Description", "AT",
                "Broker", "MatchID", "Comments", "SavedAt"]
        return _json_ok({"ok": True, "rows": _to_records(hist[cols])})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"History load error: {e}", 500)


@app.route("/history/recall", methods=["POST"])
def history_recall():
    try:
        st = _load_state()
        account = (st.get("account") or "").strip()
        if not account:
            return _json_err("Select an account first", 400)

        df = _load_df("rec.pkl")
        if df is None:
            return _json_err("No active data", 400)

        body = request.get_json(silent=True) or {}
        rows = body.get("rows") or []
        if not rows:
            return _json_err("No rows provided", 400)

        pick = pd.DataFrame(rows)
        # Extract MatchIDs (if provided)
        mids = set(
            str(m).strip()
            for m in pick.get("MatchID", pd.Series(dtype=str)).astype(str).tolist()
            if str(m).strip()
        )

        # If MatchIDs are provided, prefer group-aware recall
        if mids:
            hist = _history_load(account)

            # Check whether any of these MatchIDs already exist in the current session
            live_mask = df["MatchID"].astype(str).isin(mids)
            if live_mask.any():
                # UNMATCH IN PLACE (avoid duplicates)
                changed = 0
                for i in df.index[df.index.isin(df.index[live_mask])]:
                    if df.at[i, "OurFlag"] == "MATCHED" or df.at[i, "BrkFlag"] == "MATCHED":
                        df.at[i, "OurFlag"] = ""
                        df.at[i, "BrkFlag"] = ""
                        df.at[i, "MatchID"] = ""
                        changed += 1
                _save_df(df, "rec.pkl")

                # Remove these groups from history as well
                _history_delete_matchids(account, mids)

                return _json_ok({
                    "ok": True,
                    "mode": "in_place_unmatch",
                    "changed": int(changed),
                    "um": _df_unmatched(df)
                })

            # Else: bring the whole group(s) from history as new unmatched rows
            take = hist.loc[hist["MatchID"].astype(str).isin(mids)].copy()
            if take.empty:
                return _json_err("No matching MatchID groups found in history", 404)

            add_df = pd.DataFrame()
            add_df[COL_DATE] = pd.to_datetime(take["Date"], errors="coerce")
            add_df[COL_SYMBOL] = take["Symbol"].astype(
                str).replace(["nan", "NaN"], "")
            add_df[COL_DESC] = take["Description"].astype(
                str).replace(["nan", "NaN"], "")
            add_df[COL_AT] = pd.to_numeric(
                take["AT"], errors="coerce").fillna(0.0)
            add_df[COL_BRK] = pd.to_numeric(
                take["Broker"], errors="coerce").fillna(0.0)
            add_df["Comments"] = take.get(
                "Comments", "").astype(str)  # restore comments
            add_df["OurFlag"] = ""
            add_df["BrkFlag"] = ""
            add_df["MatchID"] = ""
            add_df = add_df.dropna(subset=[COL_DATE]).reset_index(drop=True)

            base = df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK,
                       "Comments", "OurFlag", "BrkFlag", "MatchID"]].copy()
            merged = pd.concat([base, add_df], ignore_index=True)
            merged["RowID"] = merged.index.astype(int) + 1
            _save_df(merged, "rec.pkl")

            _history_delete_matchids(account, mids)

            return _json_ok({
                "ok": True,
                "mode": "import_from_history",
                "added": int(len(add_df)),
                "um": _df_unmatched(merged)
            })

        # ---- Legacy path: no MatchID provided (row-by-row recall) ----
        add = pick.copy()
        if add.empty:
            return _json_err("No rows provided", 400)

        add_df = pd.DataFrame()
        add_df[COL_DATE] = pd.to_datetime(add.get("Date", ""), errors="coerce")
        add_df[COL_SYMBOL] = add.get("Symbol", "").astype(
            str).replace(["nan", "NaN"], "")
        add_df[COL_DESC] = add.get("Description", "").astype(
            str).replace(["nan", "NaN"], "")
        add_df[COL_AT] = pd.to_numeric(
            add.get("AT", 0.0), errors="coerce").fillna(0.0)
        add_df[COL_BRK] = pd.to_numeric(
            add.get("Broker", 0.0), errors="coerce").fillna(0.0)
        add_df["Comments"] = add.get("Comments", "").astype(
            str) if "Comments" in add else ""
        add_df["OurFlag"] = ""
        add_df["BrkFlag"] = ""
        add_df["MatchID"] = ""
        add_df = add_df.dropna(subset=[COL_DATE]).reset_index(drop=True)

        base = df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK,
                   "Comments", "OurFlag", "BrkFlag", "MatchID"]].copy()
        merged = pd.concat([base, add_df], ignore_index=True)
        merged["RowID"] = merged.index.astype(int) + 1
        _save_df(merged, "rec.pkl")

        # If legacy payload carried MatchIDs, clean those groups from history, too
        legacy_mids = set(
            str(m).strip()
            for m in add.get("MatchID", pd.Series(dtype=str)).astype(str).tolist()
            if str(m).strip()
        )
        if legacy_mids:
            _history_delete_matchids(account, legacy_mids)

        return _json_ok({"ok": True, "mode": "legacy_rows", "added": int(len(add_df)), "um": _df_unmatched(merged)})
    except Exception as e:
        traceback.print_exc()
        return _json_err(f"History recall error: {e}", 500)


# ---------- Downloads ----------
# --- helpers for safe filename parts & date normalization ---
def _normalize_date_str(s) -> str:
    """
    Return YYYY-MM-DD from common inputs:
    2025-10-11, 11/10/2025, 10/11/2025, 11-10-2025, 10-11-2025, 2025/10/11, 11 Oct 2025, Oct-11-2025, etc.
    """
    def _try_parse(txt, fmts):
        for fmt in fmts:
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                pass
        return None

    if not s or not str(s).strip():
        return datetime.now().strftime("%Y-%m-%d")

    s = str(s).strip()

    # 1) direct common formats
    fmts = (
        "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%m-%d-%Y",
        "%d %b %Y",  "%d %B %Y",
        "%b-%d-%Y",  "%B-%d-%Y",
        "%d.%m.%Y"
    )
    dt = _try_parse(s, fmts)
    if dt:
        return dt.strftime("%Y-%m-%d")

    # 2) ISO-like prefix: 2025-10-11T12:34 -> take first 10
    if len(s) >= 10 and s[4] in "-/" and s[7] in "-/":
        return s[:10].replace("/", "-")

    # 3) compact digits? e.g. 20251011 or 11102025 (ddmmyyyy / yyyymmdd)
    m = re.fullmatch(r"(\d{8})", s)
    if m:
        raw = m.group(1)
        # try yyyymmdd
        try:
            return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
        # try ddmmyyyy
        try:
            return datetime.strptime(raw, "%d%m%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass

    # fallback: today
    return datetime.now().strftime("%Y-%m-%d")


def _slug(s: str, fallback: str = "Unknown") -> str:
    """Make a filesystem-safe token (remove spaces & specials)."""
    s = (s or "").strip() or fallback
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or fallback


@app.route("/download_recon", methods=["POST"])
# Define a function that builds and downloads a reconciliation workbook as an Excel file.
def download_recon():
    # Wrap the whole routine in try/except so errors return a controlled HTTP response.
    try:
        # Load the working reconciliation DataFrame from a cached pickle file.
        df = _load_df("rec.pkl")
        # If no data is present, exit early with a 400 Bad Request and message.
        if df is None:
            return ("No active data", 400)

        # Load persisted UI/state values (account, broker, dates, balances, etc.).
        st = _load_state()
        # Create a DataFrame of currently unmatched items (the breaks to review).
        um = pd.DataFrame(_df_unmatched(df))
        um.rename(columns={"DateKey": "Date", COL_DESC: "Description"}, inplace=True)


        # Clean numeric columns
        for col in ["AT", "Broker", "Difference"]:
            if col in um.columns:
                um[col] = (
                    pd.to_numeric(um[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )


        # Allocate an in-memory buffer to assemble the Excel file (no disk I/O).
        buf = io.BytesIO()
        # Prefer the xlsxwriter engine (richer formatting), else fall back gracefully.
        try:
            # Try using xlsxwriter for fine-grained formatting control.
            writer = pd.ExcelWriter(
                buf,
                engine="xlsxwriter",
                engine_kwargs={"options": {"nan_inf_to_errors": True}},
            )
            use_xlsxwriter = True
        except Exception:
            # If xlsxwriter isn't available, switch to openpyxl (basic writing).
            writer = pd.ExcelWriter(buf, engine="openpyxl")
            # Flag that we must take the simpler path without formats.
            use_xlsxwriter = False

        # Use the writer as a context manager so it saves/flushes automatically.
        with writer as xw:
            # If xlsxwriter is available, build styled worksheets manually.
            if use_xlsxwriter:
                # Add a primary worksheet called "Recon".
                sh = xw.book.add_worksheet("Recon")
                # Register it so writer is aware of this sheet by name.
                xw.sheets["Recon"] = sh

                # Define a bold, colored, bordered header format for table headers.
                f_hdr = xw.book.add_format(
                    {"bold": True, "bg_color": "#9bd1f7", "border": 1, "align": "center"})
                # Define a bold label format for field names.
                f_lab = xw.book.add_format({"bold": True, "border": 1})
                # Define a generic value format with borders.
                f_val = xw.book.add_format({"border": 1})
                # Define a currency/number format with red negatives and borders.
                f_money = xw.book.add_format(
                    {"num_format": "#,##0.00;[Red](#,##0.00)", "border": 1})
                # Define a small italic format for helper notes.
                f_small = xw.book.add_format({"italic": True, "font_size": 9})

                # Set column B width for dates.
                sh.set_column("B:B", 12)
                # Set wider columns for symbol and description.
                sh.set_column("C:D", 28)
                # Set medium width for money columns.
                sh.set_column("E:F", 22)
                # Set width for difference and comments columns.
                sh.set_column("G:H", 18)

                # Write the "Date" label cell.
                sh.write("B2", "Date", f_lab)
                # Write the reconciliation date from state (or empty if missing).
                sh.write("C2", st.get("recon_date") or "", f_val)
                # Add a gentle hint that the date can be updated manually.
                sh.write("D2", "Manually Update", f_small)
                # Write the "Account" label cell.
                sh.write("B5", "Account", f_lab)
                # Write the account identifier/name from state.
                sh.write("C5", st.get("account") or "", f_val)

                # Define a helper that writes currency cells or leaves them blank when ~zero.
                def write_money_or_blank(ws, r, c, v):
                    """
                    Safe writer for numeric cells: never passes NaN/Inf to write_number().
                    """
                    # Try convert to float
                    try:
                        n = float(v)
                    except Exception:
                        n = 0.0

                    # Treat NaN/Inf as zero
                    if not math.isfinite(n):
                        n = 0.0

                    # If effectively zero, show blank
                    if abs(n) < 1e-9:
                        ws.write(r, c, "", f_val)
                    else:
                        ws.write_number(r, c, n, f_money)



                # Read ending balance (AT) from state with a safe numeric default.
                eb_at = float(st.get("eb_at")) if st.get(
                    "eb_at") is not None else 0.0
                # Read ending balance (Broker) from state with a safe numeric default.
                eb_brk = float(st.get("eb_brk")) if st.get(
                    "eb_brk") is not None else 0.0
                # Compute the ending balance difference (broker minus AT), rounded to cents.
                eb_diff = round(eb_brk - eb_at, 2)
                # Sum unmatched Differences to get the transaction total (robust to empties/NAs).
                txn_total = round(float(um.get("Difference", pd.Series(
                    [])).fillna(0).sum()), 2) if not um.empty else 0.0
                # Compute the final difference after applying transaction total against EB diff.
                final_diff = round(txn_total - eb_diff, 2)

                # Label and write the AT ending balance with money formatting.
                sh.write("E2", "Ending balance  AT", f_lab)
                write_money_or_blank(sh, 1, 5, eb_at)
                # Label and write the broker ending balance.
                sh.write("E3", "Ending balance broker", f_lab)
                write_money_or_blank(sh, 2, 5, eb_brk)
                # Label and write the EB difference.
                sh.write("E4", "Ending Balance Difference", f_lab)
                write_money_or_blank(sh, 3, 5, eb_diff)
                # Label and write the transaction total from unmatched items.
                sh.write("E5", "Transaction Total", f_lab)
                write_money_or_blank(sh, 4, 5, txn_total)
                # Label and write the final difference for reconciliation status.
                sh.write("E6", "Final Difference", f_lab)
                write_money_or_blank(sh, 5, 5, final_diff)

                # Choose a starting row index for the unmatched table (adds top spacing).
                start_row = 8
                # Define headers for the unmatched (Recon) detail table.
                headers = ["Date", "Symbol", "Description",
                           "AT", "Broker", "Difference", "Comments"]
                # Write table headers with header format, starting at column index 1 (column B).
                for c, h in enumerate(headers, start=2):
                    sh.write(start_row, c - 1, h, f_hdr)

                # Initialize the first data row just under the headers.
                r = start_row + 1
                # If there are unmatched rows, write them out one per row.
                if not um.empty:
                    # Iterate through each row to populate the Recon table.
                    for _, row in um.iterrows():
                        # Write the Date value (column B).
                        sh.write(r, 1, row.get("Date") or "", f_val)
                        # Write the Symbol value (column C).
                        sh.write(r, 2, row.get("Symbol") or "", f_val)
                        # Write the Description value (column D).
                        sh.write(r, 3, row.get("Description") or "", f_val)
                        # Write the AT amount (column E) with currency formatting.
                        write_money_or_blank(sh, r, 4, row.get("AT", 0.0))
                        # Write the Broker amount (column F) with currency formatting.
                        write_money_or_blank(sh, r, 5, row.get("Broker", 0.0))
                        # Write the Difference amount (column G) with currency formatting.
                        write_money_or_blank(
                            sh, r, 6, row.get("Difference", 0.0))
                        # Write any Comments (column H) as plain text.
                        sh.write(r, 7, row.get("Comments") or "", f_val)
                        # Advance to the next output row.
                        r += 1
                # Freeze panes to keep headers/left columns visible while scrolling.
                sh.freeze_panes(start_row + 1, 2)

                # Normalize the account string and load historical cleared breaks for it.
                account = (st.get("account") or "").strip()
                # Load existing persisted history for "Cleared Breaks".
                hist = _history_load(account)

                if not hist.empty:
                    for col in ["AT", "Broker"]:
                        if col in hist.columns:
                            hist[col] = (
                                pd.to_numeric(hist[col], errors="coerce")
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                            )


                    # Reload history so the sheet reflects the newly appended records.
                    hist = _history_load(account)

                # Add a second worksheet for listing previously cleared (matched) breaks.
                sh2 = xw.book.add_worksheet("Cleared Breaks")
                # Register this sheet with the writer by name.
                xw.sheets["Cleared Breaks"] = sh2
                # Set a compact date column width.
                sh2.set_column("A:A", 12)
                # Widen symbol/description columns for readability.
                sh2.set_column("B:C", 28)
                # Set widths for money columns.
                sh2.set_column("D:E", 18)
                # Set width for MatchID column.
                sh2.set_column("F:F", 16)
                # Set width for Comments column.
                sh2.set_column("G:G", 28)

                # Define the headers for the Cleared Breaks table.
                hdr2 = ["Date", "Symbol", "Description", "AT",
                        "Broker", "MatchID", "Comments", "SavedAt"]
                # Reuse a styled header format for the cleared list.
                hdr_fmt = xw.book.add_format(
                    {"bold": True, "bg_color": "#9bd1f7", "border": 1, "align": "center"})
                # Write the header row at the top (row 0).
                for c, h in enumerate(hdr2):
                    sh2.write(0, c, h, hdr_fmt)

                # Start writing history rows just under the header.
                rr = 1
                # If there is historical data, sort and write it out.
                if not hist.empty:
                    # Sort to achieve stable, meaningful ordering (mergesort is stable).
                    hist = hist.sort_values(
                        ["Date", "MatchID", "Symbol"], kind="mergesort")
                    # Populate each row of the Cleared Breaks sheet.
                    for _, row in hist.iterrows():
                        # Write the Date value.
                        sh2.write(rr, 0, row.get("Date", ""), f_val)
                        # Write the Symbol string.
                        sh2.write(rr, 1, str(
                            row.get("Symbol", "")) or "", f_val)
                        # Write the Description string.
                        sh2.write(rr, 2, str(
                            row.get("Description", "")) or "", f_val)
                        # Write the AT amount using money-or-blank logic.
                        write_money_or_blank(sh2, rr, 3, row.get("AT", 0.0))
                        # Write the Broker amount similarly.
                        write_money_or_blank(
                            sh2, rr, 4, row.get("Broker", 0.0))
                        # Write the MatchID (as string to avoid numeric formatting).
                        sh2.write(rr, 5, str(
                            row.get("MatchID", "")) or "", f_val)
                        # Write any Comments text.
                        sh2.write(rr, 6, str(
                            row.get("Comments", "")) or "", f_val)
                        # Write the SavedAt timestamp text.
                        sh2.write(rr, 7, str(
                            row.get("SavedAt", "")) or "", f_val)
                        # Advance to the next row.
                        rr += 1
            else:
                # If we only have openpyxl, export simple sheets without custom styling.
                um.to_excel(xw, index=False, sheet_name="Recon")
                # Also export the cleared history for the account to its own sheet.
                _history_load((st.get("account") or "").strip()).to_excel(
                    xw, index=False, sheet_name="Cleared Breaks")

        # Reset the buffer position so Flask's send_file reads from the start.
        buf.seek(0)
        # Build a YYYYMMDD tag from the recon_date (or today's date) for the filename.
        date_tag = (st.get("recon_date") or datetime.now().strftime(
            "%Y-%m-%d")).replace("-", "")
        # Derive a safe Broker Name for the filename; try multiple keys, then default.
        broker_name = _safe_account_name(
            st.get("broker_name") or st.get("broker") or "Broker")
        # Derive a safe Account Number for the filename; fall back to 'account' if needed.
        account_num = _safe_account_name(
            st.get("account_number") or st.get("account") or "Account")
        # Construct the requested filename format: BrokerName_AccountNumber_ReconDate.xlsx
        out_name = f"{broker_name}_{account_num}_{date_tag}.xlsx"
        # Send the in-memory workbook to the client as a downloadable Excel attachment.
        return send_file(
            buf,
            as_attachment=True,
            download_name=out_name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    # Handle any unexpected exceptions to avoid crashing the request.
    except Exception as e:
        # Log the full traceback to aid debugging on the server side.
        traceback.print_exc()
        # Return a 500 Internal Server Error with the message for visibility.
        return ("Recon export error: " + str(e), 500)


@app.route("/download_report", methods=["GET"])
def download_report():
    try:
        df = _load_df("rec.pkl")
        if df is None:
            return ("No active data", 400)

        buf = io.BytesIO()
        try:
            writer = pd.ExcelWriter(buf, engine="xlsxwriter")
        except Exception:
            writer = pd.ExcelWriter(buf, engine="openpyxl")

        with writer as xw:
            df2 = df.copy()
            df2["Difference"] = (df2[COL_BRK] - df2[COL_AT]).round(2)
            df2.to_excel(xw, index=False, sheet_name="All_Rows")
            pd.DataFrame(_df_matched(df)).to_excel(
                xw, index=False, sheet_name="Matched")
            pd.DataFrame(_df_unmatched(df)).to_excel(
                xw, index=False, sheet_name="Unmatched")
        buf.seek(0)
        out_name = f"matched_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buf,
            as_attachment=True,
            download_name=out_name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        traceback.print_exc()
        return ("Report export error: " + str(e), 500)


@app.route("/reset", methods=["POST"], endpoint="reset_session")
def reset_session():
    sid = session.get("sid")
    if sid:
        try:
            shutil.rmtree(TMP_ROOT / sid, ignore_errors=True)
        except Exception:
            pass
    session.pop("sid", None)
    return redirect(url_for("index"))

# --------------------------------------------------------------------------------------
# Run (works for python + PyInstaller EXE)
# --------------------------------------------------------------------------------------


def _is_listening(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.25)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _open_browser_when_ready():
    for _ in range(60):  # ~12 seconds total
        if _is_listening(HOST, PORT):
            try:
                webbrowser.open_new(URL)
            except Exception:
                try:
                    os.startfile(URL)  # Windows fallback
                except Exception:
                    pass
            return
        time.sleep(0.2)
    try:
        webbrowser.open_new(URL)
    except Exception:
        try:
            os.startfile(URL)
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Only open browser if running locally (not in production/Railway)
    is_production = _IS_CONTAINER
    if not is_production:
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    
    app.run(host=HOST, port=PORT, debug=False,
            use_reloader=False, threaded=True)