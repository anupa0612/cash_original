# brokers/gtna.py
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


# -----------------------------
# Normalization helpers
# -----------------------------

REQ_CANON = ["transdate", "valuedate", "description", "debit", "credit"]

_DATE_INPUT_FMT = "%d/%m/%Y"  # dd/mm/yyyy as specified


def _canon(s: str) -> str:
    """Normalize a header token: lowercase, remove spaces/dots and non-letters."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")  # BOM
    s = s.strip().lower()
    s = re.sub(r"[^a-z]", "", s)  # keep only letters
    return s


def _parse_date_ddmmyyyy(s: str) -> Optional[pd.Timestamp]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, _DATE_INPUT_FMT)
        return pd.to_datetime(dt)
    except Exception:
        # allow pandas to try (dayfirst true) but still expect DD/MM/YYYY
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return None if pd.isna(dt) else dt
    
def _parse_user_date(s: str) -> Optional[pd.Timestamp]:
    """
    Parse date coming from the UI.
    Try common formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return pd.to_datetime(dt)
        except Exception:
            continue

    # last resort: let pandas guess
    dt = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(dt) else dt



def _to_number(val) -> float:
    """
    Robust numeric parser:
    - removes commas, NBSPs, thin spaces, currency codes, and any non-numeric except dot/minus
    - normalizes Unicode minus to ASCII '-'
    - returns 0.0 on failure
    """
    if val is None:
        return 0.0
    s = str(val)

    # common invisible / spacing chars
    s = s.replace("\u00A0", " ")   # NBSP -> space
    s = s.replace("\u2009", " ")   # thin space
    s = s.replace("\u2212", "-")   # Unicode minus → ASCII minus
    s = s.strip()

    if s == "":
        return 0.0

    # remove thousand separators and obvious currency/unit text
    s = s.replace(",", "")
    # keep only digits, dot, minus (strip everything else like 'USD', spaces, etc.)
    s = re.sub(r"[^0-9.\-]", "", s)

    # normalize minus and multiple dots defensively
    s = re.sub(r"-+", "-", s)
    if s.count("-") > 1:
        s = "-" + s.replace("-", "")
    if s.count(".") > 1:
        first, *rest = s.split(".")
        s = first + "." + "".join(rest)

    try:
        return float(s)
    except Exception:
        return 0.0


DIV_RX = re.compile(r"cash\s*dividend\s*:\s*([^(]+?)\s*\(", re.I)


def _extract_div_symbol(desc: str) -> str:
    """
    From text like:
      'Cash dividend : AAPY (XS2901884663) - dividend per share : 0.2220588 USD'
    extract 'AAPY' (but can be any length, not always 4 chars).
    """
    if not isinstance(desc, str):
        return ""
    m = DIV_RX.search(desc)
    if not m:
        return ""
    return m.group(1).strip().upper()


def _looks_trade(desc: str) -> bool:
    """
    Treat a line as part of exchange settlements if:
      - Description starts with 'Buy ' or 'Sell ', OR
      - Description contains the word 'Adjustment' (anywhere, any case).
    """
    if not isinstance(desc, str):
        return False

    d = desc.strip().lower()

    # Normal trades
    if d.startswith("buy ") or d.startswith("sell "):
        return True

    # Also pull adjustment lines into the same TD aggregation
    if "adjustment" in d:
        return True
    
    if "stamp duty" in d:
        return True

    return False



# -----------------------------
# CSV reading
# -----------------------------

def _find_header_row(df_raw: pd.DataFrame) -> Tuple[int, dict]:
    """
    Find the row index that contains the header with required columns.
    Return (header_row_index, mapping_dict) where mapping maps canonical name -> actual column name.
    Raises ValueError if not found.
    """
    max_scan = min(50, len(df_raw))  # scan a bit deeper to be safe
    for r in range(max_scan):
        row = df_raw.iloc[r].astype(str).tolist()
        cands = [_canon(x) for x in row]
        mapping = {}
        for need in REQ_CANON:
            if need in cands:
                idx = cands.index(need)
                mapping[need] = str(df_raw.iloc[r, idx]).strip()
            else:
                mapping[need] = None

        if all(mapping[k] for k in REQ_CANON):
            return r, mapping

    raise ValueError(
        "GTNA CSV: header with Trans.Date/Value Date/Description/Debit/Credit not found"
    )


def _clean_one_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw DataFrame read without headers, detect header row, rename,
    select required columns, and return a normalized table.
    """
    if df_raw.empty:
        return pd.DataFrame(columns=["SettleDate", "ReportDate", "Symbol", "Description", "Amount", "TradeDate"])

    # Find header row
    hdr_row, mapping = _find_header_row(df_raw)

    # Slice body and set columns to the discovered header row
    body = df_raw.iloc[hdr_row + 1:].copy()
    body.columns = df_raw.iloc[hdr_row].astype(str).tolist()

    # Keep required columns (using actual header names found)
    cols = [
        mapping["transdate"],
        mapping["valuedate"],
        mapping["description"],
        mapping["debit"],
        mapping["credit"],
    ]
    out = body[cols].copy()
    out.columns = ["Trans.Date", "Value Date", "Description", "Debit", "Credit"]

    # Drop fully empty rows and the "Opening Balance" line
    out = out.dropna(how="all")
    out["Description"] = out["Description"].astype(str).fillna("")
    out = out[out["Description"].str.strip().str.lower() != "opening balance"]

    # Keep only rows with date-like values in both date fields (per your rule)
    out["Trans.Date"] = out["Trans.Date"].astype(str)
    out["Value Date"] = out["Value Date"].astype(str)
    out["TransDateParsed"] = out["Trans.Date"].map(_parse_date_ddmmyyyy)
    out["ValueDateParsed"] = out["Value Date"].map(_parse_date_ddmmyyyy)
    out = out[out["TransDateParsed"].notna() & out["ValueDateParsed"].notna()]

    # Amounts (robust)
    out["Debit"] = out["Debit"].map(_to_number).fillna(0.0)
    out["Credit"] = out["Credit"].map(_to_number).fillna(0.0)
    out["Amount"] = (out["Credit"] - out["Debit"]).astype(float)  # Signed amount

    # Final normalization columns
    out["SettleDate"] = out["ValueDateParsed"]   # filter by Value Date
    out["ReportDate"] = out["TransDateParsed"]   # display/rec date is Trans.Date
    out["Symbol"] = out["Description"].map(_extract_div_symbol)  # dividends only

    # Keep canonical order + TradeDate (for exchange aggregation)
    out = out[["SettleDate", "ReportDate", "Symbol", "Description", "Amount", "TransDateParsed"]]
    out = out.rename(columns={"TransDateParsed": "TradeDate"})
    out = out.reset_index(drop=True)

    # If truly empty, return empty frame with correct columns
    if out.empty:
        return pd.DataFrame(columns=["SettleDate", "ReportDate", "Symbol", "Description", "Amount", "TradeDate"])

    return out


def _load_one_csv(raw: bytes) -> pd.DataFrame:
    """
    Read a single CSV (possibly with notes before header), normalize it.
    """
    df_raw = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        dtype=str,
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    return _clean_one_dataframe(df_raw)


def _iter_zip_csvs(raw: bytes):
    """Yield (name, bytes) for each .csv file inside a zip."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for zi in z.infolist():
            if zi.is_dir():
                continue
            if not zi.filename.lower().endswith(".csv"):
                continue
            yield zi.filename, z.read(zi)


def _load_csvs_from_upload(file_storage) -> List[pd.DataFrame]:
    """
    Accepts:
      - a single CSV
      - a .zip containing multiple CSVs
    Returns a list of normalized dataframes (some may be empty; we filter later).
    """
    raw = file_storage.read()
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    filename = (getattr(file_storage, "filename", "") or "").lower()

    frames: List[pd.DataFrame] = []
    if filename.endswith(".zip"):
        any_file = False
        for name, blob in _iter_zip_csvs(raw):
            any_file = True
            try:
                df_part = _load_one_csv(blob)
            except Exception:
                df_part = pd.DataFrame(
                    columns=["SettleDate", "ReportDate", "Symbol", "Description", "Amount", "TradeDate"]
                )
            frames.append(df_part)
        if not any_file:
            raise ValueError("GTNA ZIP: no CSV files found")
    else:
        frames.append(_load_one_csv(raw))

    # Keep only non-empty frames
    frames = [f for f in frames if not f.empty]
    return frames


# -----------------------------
# Exchange settlements aggregator
# -----------------------------

def _apply_exchange_settlements(df: pd.DataFrame) -> pd.DataFrame:
    """
    From df (already filtered by date range), aggregate rows whose Description starts
    with 'Buy ' or 'Sell ' by TradeDate. Keep other rows unchanged.
    Replace each day's trade lines with one summary row:
      Description = 'Exchange Settlements - TD dd.mm.yyyy'
      Symbol = '' (blank)
      SettleDate = NaT (not used later)
      ReportDate = that TradeDate
      Amount = sum of signed amounts for that TradeDate
    """
    if df.empty:
        return df

    mask_trade = df["Description"].astype(str).map(_looks_trade)
    trades = df.loc[mask_trade].copy()
    rest = df.loc[~mask_trade].copy()

    if trades.empty:
        return df

    grp = (
        trades.groupby("TradeDate", dropna=False)["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"Amount": "Amt"})
    )

    rows = []
    for _, r in grp.iterrows():
        tdate = r["TradeDate"]  # Timestamp
        amt = float(r["Amt"]) if pd.notna(r["Amt"]) else 0.0
        title = f"Exchange Settlements - TD {tdate.strftime('%d.%m.%Y')}" if pd.notna(tdate) else "Exchange Settlements"
        rows.append(
                {
                    # Use TradeDate as the effective date for the merged exchange settlement line
                    "SettleDate": tdate,
                    "ReportDate": tdate,
                    "Symbol": "",
                    "Description": title,
                    "Amount": amt,
                    "TradeDate": tdate,
                }
            )
    agg_df = pd.DataFrame(rows, columns=["SettleDate", "ReportDate", "Symbol", "Description", "Amount", "TradeDate"])
    out = pd.concat([rest, agg_df], ignore_index=True)
    return out


# -----------------------------
# Public API
# -----------------------------

def clean_broker(file_storage, start_date_str: str, end_date_str: str, account_value: str = None) -> pd.DataFrame:
    """
    Build normalized GTNA broker dataframe:
      Columns: SettleDate, ReportDate, Symbol, Description, Amount
    Filtering:
      - Keep only rows with SettleDate in [start_date, end_date] (inclusive).
      - Then apply exchange-settlement aggregation on rows that start with Buy/Sell by TradeDate.
    """
    sdt = _parse_user_date(str(start_date_str))
    edt = _parse_user_date(str(end_date_str))
    if sdt is None or edt is None or sdt > edt:
        raise ValueError("Invalid date range. Please check From / To dates.")

    frames = _load_csvs_from_upload(file_storage)

    if not frames:
        raise ValueError("GTNA: no transactions found in the provided file(s).")

    # Merge all CSVs
    bro = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Filter by SettleDate (Value Date)
    bro = bro.dropna(subset=["SettleDate"]).copy()
    bro = bro[(bro["SettleDate"] >= sdt) & (bro["SettleDate"] <= edt)].reset_index(drop=True)

    # Apply exchange settlements AFTER date filtering
    bro = _apply_exchange_settlements(bro)

    # Final tidy
    bro = bro[["SettleDate", "ReportDate", "Symbol", "Description", "Amount"]].copy()
    bro["Amount"] = pd.to_numeric(bro["Amount"], errors="coerce").fillna(0.0)
    bro = bro[(bro["Description"].astype(str).str.strip() != "") | (bro["Amount"].abs() > 1e-9)]
    bro = bro.reset_index(drop=True)
    return bro


def build_rec(at_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the normalized broker_df into reconciliation rows and append AT side.
    Output columns: Date, Symbol, Description, AT, Broker, Comments, RowID, OurFlag, BrkFlag, MatchID
    """
    COL_DATE, COL_DESC, COL_AT, COL_BRK, COL_SYMBOL = "Date", "Description", "AT", "Broker", "Symbol"

    if broker_df is not None and not broker_df.empty:
        brk = pd.DataFrame(
            {
                COL_DATE: broker_df["SettleDate"],
                COL_SYMBOL: broker_df.get("Symbol", "").astype(str),
                COL_DESC: broker_df["Description"].astype(str),
                COL_AT: 0.0,
                COL_BRK: broker_df["Amount"].astype(float),
                "Comments": "",
            }
        )
    else:
        brk = pd.DataFrame(
            {
                COL_DATE: pd.Series([], dtype="datetime64[ns]"),
                COL_SYMBOL: pd.Series([], dtype=str),
                COL_DESC: pd.Series([], dtype=str),
                COL_AT: pd.Series([], dtype="float64"),
                COL_BRK: pd.Series([], dtype="float64"),
                "Comments": pd.Series([], dtype=str),
            }
        )

    rec = _finalize(brk)
    rec = _append_at(rec, at_df, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK)
    rec[COL_AT] = pd.to_numeric(rec[COL_AT], errors="coerce").fillna(0.0)
    rec[COL_BRK] = pd.to_numeric(rec[COL_BRK], errors="coerce").fillna(0.0)
    return rec


# ---- finalize helpers (same shape as other brokers) ----

def _finalize(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        base = pd.DataFrame(
            {
                "Date": pd.Series([], dtype="datetime64[ns]"),
                "Symbol": pd.Series([], dtype=str),
                "Description": pd.Series([], dtype=str),
                "AT": pd.Series([], dtype="float64"),
                "Broker": pd.Series([], dtype="float64"),
                "Comments": pd.Series([], dtype=str),
            }
        )
        base["RowID"] = pd.Series([], dtype="int64")
        base["OurFlag"] = pd.Series([], dtype=str)
        base["BrkFlag"] = pd.Series([], dtype=str)
        base["MatchID"] = pd.Series([], dtype=str)
        return base

    rec = rec.sort_values(["Date", "Symbol", "Description"], kind="mergesort").reset_index(drop=True)
    rec["RowID"] = rec.index.astype(int) + 1
    rec["OurFlag"] = ""
    rec["BrkFlag"] = ""
    rec["MatchID"] = ""
    rec["DateKey"] = rec["Date"].dt.strftime("%Y-%m-%d").fillna("")
    rec["Difference"] = (rec["Broker"] - rec["AT"]).round(2)
    return rec


def _append_at(rec: pd.DataFrame, at_df: pd.DataFrame, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK) -> pd.DataFrame:
    if at_df is None or at_df.empty:
        return rec
    at_part = at_df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]]
    base = rec.drop(columns=["RowID", "OurFlag", "BrkFlag", "MatchID", "DateKey", "Difference"], errors="ignore")
    merged = pd.concat([base, at_part], ignore_index=True)
    return _finalize(merged)
