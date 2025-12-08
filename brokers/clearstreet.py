# brokers/clearstreet.py
import io
import re
import pandas as pd

"""
Internal broker_df schema expected by app.py:
  SettleDate (datetime64)
  ReportDate (datetime64)
  Symbol     (str)
  Desc       (str)
  Amount     (float)    # already signed correctly for Clear Street

Exposes:
  clean_broker(file_storage, start_date_str, end_date_str) -> broker_df
  build_rec(at_df, broker_df) -> standardized reconciliation rows
"""

# -------------------------
# Helpers
# -------------------------
_CURRENCY_RX = re.compile(r"[,$₹£€]")
_PARENS_NEG  = re.compile(r"\(([^)]+)\)")
_UNICODE_MINUS = "−"  # U+2212

def _clean_amount(series: pd.Series) -> pd.Series:
    # IMPORTANT: use vectorized string ops on Series
    s = series.astype(str)
    s = s.str.replace(_UNICODE_MINUS, "-", regex=False)
    s = s.str.replace(_CURRENCY_RX, "", regex=True)          # <-- fixed
    s = s.str.replace(_PARENS_NEG, r"-\1", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _norm_header(h: str) -> str:
    h = (h or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", h)

def _pick_header(cols_norm: dict, *candidates: str) -> str | None:
    for cand in candidates:
        k = _norm_header(cand)
        if k in cols_norm:
            return cols_norm[k]
    return None

def _is_option_symbol(sym: str) -> bool:
    if not isinstance(sym, str):
        return False
    s = sym.strip().replace(" ", "")
    return len(s) > 10

# -------------------------
# Public API
# -------------------------
def clean_broker(file_storage, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date_str, errors="coerce")
    end_dt   = pd.to_datetime(end_date_str,   errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("Invalid date range. Please check From / To dates.")

    data = file_storage.read()
    file_storage.stream.seek(0)
    fname = (file_storage.filename or "").lower()

    if fname.endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    else:
        raw = pd.read_excel(io.BytesIO(data), dtype=str)

    if raw.empty:
        return pd.DataFrame(columns=["SettleDate", "ReportDate", "Symbol", "Desc", "Amount"])

    cols_norm = {_norm_header(c): c for c in raw.columns}

    c_settle = _pick_header(cols_norm, "settlement date", "settle date", "settledate", "settlement_dt", "settle_dt")
    c_trade  = _pick_header(cols_norm, "trade date", "tradedate", "trade_dt", "report date", "reportdate")
    c_sym    = _pick_header(cols_norm, "symbol", "ticker")
    c_desc   = _pick_header(cols_norm, "description", "dispdescr", "details")
    c_amt    = _pick_header(cols_norm, "net amount", "netamount", "amount", "n_amt", "net amount (usd)")

    missing = [n for n, c in {
        "Settlement Date": c_settle,
        "Trade Date":      c_trade,
        "Symbol":          c_sym,
        "Description":     c_desc,
        "Net Amount":      c_amt,
    }.items() if c is None]
    if missing:
        raise ValueError(f"Clear Street file missing required columns: {', '.join(missing)}")

    settle_dt = pd.to_datetime(raw[c_settle], errors="coerce", dayfirst=False)
    report_dt = pd.to_datetime(raw[c_trade],  errors="coerce", dayfirst=False)
    amt       = _clean_amount(raw[c_amt])

    out = pd.DataFrame({
        "SettleDate": settle_dt,
        "ReportDate": report_dt,
        "Symbol": raw[c_sym].astype(str).fillna("").str.strip().str.upper(),
        "Desc":   raw[c_desc].astype(str).fillna("").str.strip(),
        "Amount": amt,  # Clear Street already signed
    })

    nonzero  = out["Amount"] != 0.0
    in_range = (out["SettleDate"] >= start_dt) & (out["SettleDate"] <= end_dt)
    out = out.loc[nonzero & in_range].dropna(subset=["SettleDate"]).reset_index(drop=True)
    return out

def build_rec(at_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    COL_DATE, COL_DESC, COL_AT, COL_BRK, COL_SYMBOL = "Date", "Description", "AT", "Broker", "Symbol"

    # empty broker case
    if broker_df is None or broker_df.empty:
        rec = _finalize(pd.DataFrame(columns=[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]))
        return _append_at(rec, at_df, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK)

    # ---- identify trades (Bought/Sold) ----
    desc = broker_df["Desc"].astype(str)
    is_trade = desc.str.match(r"(?i)^\s*(bought|sold)\b", na=False)

    # ---- normalize the date we will USE in the rec = TRADE DATE, fallback to settle ----
    # (we still filtered by SettleDate earlier in clean_broker)
    rec_date_trade = broker_df["ReportDate"]
    rec_date = rec_date_trade.where(rec_date_trade.notna(), broker_df["SettleDate"])

    # =========================
    # Exchange Settlements (by TRADE DATE)
    # =========================
    exch = broker_df.loc[is_trade].copy()
    if not exch.empty:
        # pick the date for grouping: Trade Date (fallback to SettleDate if NaT)
        datekey = exch["ReportDate"].where(exch["ReportDate"].notna(), exch["SettleDate"]).dt.strftime("%Y-%m-%d")

        # equity vs options bucket
        buckets = [
            "Exchange Settlements - Options" if _is_option_symbol(s) else "Exchange Settlements - Equity"
            for s in exch["Symbol"].astype(str)
        ]

        exch = exch.assign(_Bucket=buckets, _DateKey=datekey)

        # group by TRADE DATE & bucket so trades on 2025-10-08 and 2025-10-09 produce separate rows
        g = exch.groupby(["_DateKey", "_Bucket"], dropna=False)["Amount"].sum()

        exch_rows = [{
            COL_DATE:  pd.to_datetime(datekey, errors="coerce"),  # ← Trade Date on the row
            COL_SYMBOL: "",
            COL_DESC: bucket,                                      # Equity / Options
            COL_AT:   0.0,
            COL_BRK:  float(amt),
            "Comments": "",                                        # (optional) add settle hint below
        } for (datekey, bucket), amt in g.items()]

        exch_df = pd.DataFrame(exch_rows)
    else:
        exch_df = pd.DataFrame(columns=[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"])

    # =========================
    # Non-trade broker rows (fees, divs, etc.) — also use TRADE DATE on the rec
    # =========================
    non_trade = broker_df.loc[~is_trade].copy()
    if not non_trade.empty:
        # rec date = Trade Date fallback Settle
        nt_date = non_trade["ReportDate"].where(non_trade["ReportDate"].notna(), non_trade["SettleDate"])
        nt = pd.DataFrame({
            COL_DATE:   nt_date,
            COL_SYMBOL: non_trade["Symbol"].fillna("").astype(str),
            COL_DESC:   non_trade["Desc"].astype(str),
            COL_AT:     0.0,
            COL_BRK:    non_trade["Amount"],
            "Comments": "",
        })
    else:
        nt = pd.DataFrame(columns=[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"])

    # stitch broker parts
    parts = [df for df in (exch_df, nt) if not df.empty]
    rec = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]
    )

    # finalize broker side then append AT rows
    rec = _finalize(rec)
    return _append_at(rec, at_df, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK)


# -------------------------
# Internal shape helpers
# -------------------------
def _finalize(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        rec = pd.DataFrame(columns=["Date", "Symbol", "Description", "AT", "Broker", "Comments"])
    rec = rec.sort_values(["Date", "Symbol", "Description"], kind="mergesort").reset_index(drop=True)
    rec["RowID"]   = rec.index.astype(int) + 1
    rec["OurFlag"] = ""
    rec["BrkFlag"] = ""
    rec["MatchID"] = ""
    return rec

def _append_at(rec: pd.DataFrame, at_df: pd.DataFrame, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK) -> pd.DataFrame:
    if at_df is None or at_df.empty:
        return rec
    need_cols = [COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]
    base = rec[need_cols].copy()
    at_part = at_df[need_cols].copy()
    merged = pd.concat([base, at_part], ignore_index=True)
    return _finalize(merged)
