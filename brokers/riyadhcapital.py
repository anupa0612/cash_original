# brokers/riyadhcapital.py
from __future__ import annotations
import io
import re
from datetime import datetime
from typing import List, Optional
import pandas as pd

try:
    import pdfplumber
except Exception as e:
    raise RuntimeError(
        "pdfplumber is required for Riyadh Capital PDF parsing. Install: pip install pdfplumber"
    ) from e


# ---- helpers ----
_DATE_FMTS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"]
_REF_RX = re.compile(r"\b([A-Z]\d{6,12})\b")  # e.g., E001283735


def _extract_ref(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = _REF_RX.search(s.upper())
    return m.group(1) if m else ""


def _parse_date(s: str) -> Optional[pd.Timestamp]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # normalise multiple spaces
    s = re.sub(r"\s+", " ", s)
    for fmt in _DATE_FMTS:
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except Exception:
            pass
    # last resort: let pandas guess
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return None if pd.isna(dt) else dt


def _to_float(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip()
    if s == "":
        return 0.0
    # handle (1,234.56) as negative
    m = re.fullmatch(r"\(([^)]+)\)", s)
    if m:
        s = "-" + m.group(1)
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _extract_tables_from_pdf(raw: bytes) -> list[pd.DataFrame]:
    out = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables or []:
                if not t or len(t) < 2:
                    continue
                out.append(pd.DataFrame(t))
    return out


def _normalise_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    headers = [str(x).strip().lower() for x in df.iloc[0].tolist()]
    body = df.iloc[1:].reset_index(drop=True)
    body.columns = headers

    def _pick(names, *cands):
        cands = [c.lower() for c in cands]
        for c in names:
            c = str(c).strip().lower()
            if c in cands:
                return c
        for c in names:
            c = str(c).strip().lower()
            if any(k in c for k in cands):
                return c
        return None

    c_date = _pick(headers, "trans. date", "transaction date", "date")
    c_desc = _pick(headers, "trans. details", "description", "details")
    c_deb = _pick(headers, "debit")
    c_cre = _pick(headers, "credit")
    c_bal = _pick(headers, "balance")
    c_ref = _pick(headers, "reference", "ref")  # reference column if present

    if c_date is None or c_desc is None:
        return None

    out = pd.DataFrame(
        {
            "SettleDate": body[c_date].astype(str),
            "Desc": body[c_desc].astype(str),
            "Debit": body[c_deb] if c_deb in body.columns else "",
            "Credit": body[c_cre] if c_cre in body.columns else "",
            "Balance": body[c_bal] if c_bal in body.columns else "",
        }
    ).fillna("")

    # Amount and dates
    out["Amount"] = out.apply(
        lambda r: _to_float(r.get("Credit", "")) - _to_float(r.get("Debit", "")),
        axis=1,
    )
    out["SettleDate"] = out["SettleDate"].apply(lambda s: _parse_date(str(s)))
    out["ReportDate"] = out["SettleDate"]
    out["TradeDate"] = out["SettleDate"]  # currently use SettleDate as TradeDate

    # ---- Symbol = Reference (if column exists) else try to extract from Desc ----
    if c_ref and c_ref in body.columns:
        sym = body[c_ref].astype(str).fillna("").map(lambda x: _extract_ref(x) or x.strip())
    else:
        sym = out["Desc"].map(_extract_ref)
    out["Symbol"] = sym.fillna("")

    return out[["SettleDate", "ReportDate", "TradeDate", "Symbol", "Desc", "Amount"]]


def _is_exchange_ref(sym: str) -> bool:
    """Riyadh: treat references starting with '12' as exchange-trade cash lines."""
    return isinstance(sym, str) and sym.strip().startswith("12")


def _apply_exchange_settlements(out: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exchange settlements by TradeDate for Riyadh Capital:
    - Pick rows whose Symbol (reference) starts with '12'
    - Group them by TradeDate and sum Amount
    - Replace detailed rows with one summary:
        Desc = 'Exchange Settlements - TD dd.mm.yyyy'
        SettleDate = NaT
        ReportDate = TradeDate
        Symbol = ''
    """
    if out.empty or "TradeDate" not in out.columns:
        return out

    mask_exch = out["Symbol"].astype(str).map(_is_exchange_ref)
    trades = out.loc[mask_exch].copy()
    rest = out.loc[~mask_exch].copy()

    if trades.empty:
        return out

    grp = (
        trades.groupby("TradeDate", dropna=False)["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"Amount": "Amt"})
    )

    rows = []
    for _, r in grp.iterrows():
        tdate = r["TradeDate"]
        amt = float(r["Amt"]) if pd.notna(r["Amt"]) else 0.0
        if pd.notna(tdate):
            title = f"Exchange Settlements - TD {tdate.strftime('%d.%m.%Y')}"
        else:
            title = "Exchange Settlements"
        rows.append(
            {
                "SettleDate": tdate,      # use TradeDate as the "Date" for rec
                "ReportDate": tdate,
                "TradeDate":  tdate,
                "Symbol": "",
                "Desc": title,
                "Amount": amt,
            }
        )

    agg_df = pd.DataFrame(
        rows,
        columns=["SettleDate", "ReportDate", "TradeDate", "Symbol", "Desc", "Amount"],
    )
    return pd.concat([rest, agg_df], ignore_index=True)


def clean_broker(
    file_storage, start_date_str: str, end_date_str: str, account_value: str = None
) -> pd.DataFrame:
    """
    Read Riyadh Capital PDF and return a normalized broker df:
      columns (internal) = SettleDate, ReportDate, TradeDate, Symbol, Desc, Amount
    This version also merges description lines that spill onto the next page and
    aggregates exchange settlements by TradeDate for references starting with '12'.
    """
    start_dt = pd.to_datetime(start_date_str, errors="coerce")
    end_dt = pd.to_datetime(end_date_str, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("Invalid date range. Please check From / To dates.")

    raw = file_storage.read()
    file_storage.stream.seek(0)

    # 1) extract all tables from the PDF
    tables = _extract_tables_from_pdf(raw)

    # 2) normalise each table (page)
    frames: List[pd.DataFrame] = []
    for t in tables:
        try:
            norm = _normalise_table(t)  # returns: SettleDate, ReportDate, TradeDate, Symbol, Desc, Amount
            if norm is not None:
                frames.append(norm)
        except Exception:
            # ignore tables we can't normalize
            pass

    # 3) if nothing usable from tables, try a very loose text fallback
    if not frames:
        text_rows = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                for line in txt.splitlines():
                    m = re.search(
                        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}).*?([A-Za-z].*?)\s+([()0-9,.\-]+)\s+([()0-9,.\-]+)\s+([()0-9,.\-]+)?$",
                        line,
                    )
                    if not m:
                        continue
                    dt = _parse_date(m.group(1))
                    desc = m.group(2).strip()
                    debit = _to_float(m.group(3))
                    credit = _to_float(m.group(4))
                    amount = credit - debit
                    # TradeDate = SettleDate here as well
                    text_rows.append([dt, dt, dt, "", desc, amount])

        out = pd.DataFrame(
            text_rows,
            columns=["SettleDate", "ReportDate", "TradeDate", "Symbol", "Desc", "Amount"],
        )
    else:
        # 4) merge continuation lines across pages
        merged_rows = []
        carry = None  # the last real row we added

        # helper: decide if a row is a continuation (no date/symbol, zero amount, has text)
        def is_cont_row(r) -> bool:
            no_date = (pd.isna(r["SettleDate"]) or str(r["SettleDate"]).strip() == "")
            no_symbol = (str(r.get("Symbol", "")).strip() == "")
            near_zero_amt = abs(float(r.get("Amount", 0.0) or 0.0)) < 1e-9
            has_text = bool(str(r.get("Desc", "")).strip())
            return no_date and no_symbol and near_zero_amt and has_text

        for page_df in frames:
            for _, r in page_df.iterrows():
                if is_cont_row(r) and carry is not None:
                    # append the extra text to the previous row
                    carry["Desc"] = (carry["Desc"] + "\n" + str(r["Desc"]).strip()).strip()
                    continue

                # start a new real row
                row = {
                    "SettleDate": r["SettleDate"],
                    "ReportDate": r["ReportDate"],
                    "TradeDate": r.get("TradeDate", r["SettleDate"]),
                    "Symbol": str(r.get("Symbol", "")).strip(),
                    "Desc": str(r.get("Desc", "")).strip(),
                    "Amount": float(r.get("Amount", 0.0) or 0.0),
                }
                merged_rows.append(row)
                carry = merged_rows[-1]

        out = pd.DataFrame(
            merged_rows,
            columns=["SettleDate", "ReportDate", "TradeDate", "Symbol", "Desc", "Amount"],
        )

    # 5) final clean + date filtering
    out = out.dropna(subset=["SettleDate"]).copy()
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0)
    out = out.loc[out["Amount"].abs() > 1e-9]

    # filter by SettleDate range first
    mask = (out["SettleDate"] >= start_dt) & (out["SettleDate"] <= end_dt)
    out = out.loc[mask].reset_index(drop=True)

    # apply exchange settlement aggregation by TradeDate for refs starting with '12'
    out = _apply_exchange_settlements(out)

    # final columns for the rest of the app (no TradeDate exposed outside)
    out = out[["SettleDate", "ReportDate", "Symbol", "Desc", "Amount"]].copy()
    return out


# ---------- build_rec (same shape as other brokers) ----------
def build_rec(at_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    COL_DATE, COL_DESC, COL_AT, COL_BRK, COL_SYMBOL = "Date", "Description", "AT", "Broker", "Symbol"

    if broker_df is not None and not broker_df.empty:
        brk = pd.DataFrame(
            {
                COL_DATE: broker_df["SettleDate"],
                COL_SYMBOL: broker_df.get("Symbol", ""),
                COL_DESC: broker_df["Desc"].astype(str),
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


# ===========================
# --- Riyadh Auto-Matching ---
# ===========================

# --- helpers ---
_DIV_WORDS = re.compile(r"(?i)\b(div|dividend|cash\s*div|div\s*tax|withholding|wht|wtax)\b")


def _r_round2(x):
    import pandas as pd
    return float(pd.Series([x]).round(2).iloc[0])


def _r_side(at_val, brk_val):
    a = _r_round2(abs(float(at_val)))
    b = _r_round2(abs(float(brk_val)))
    if a > 0 and b == 0:
        return "our"  # AT-only
    if b > 0 and a == 0:
        return "brk"  # Broker-only
    if a == 0 and b == 0:
        return "empty"
    return "mixed"


def _r_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DateKey"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Diff"] = (out["Broker"] - out["AT"]).apply(_r_round2)
    out["__Side"] = out.apply(lambda r: _r_side(r["AT"], r["Broker"]), axis=1)
    out["OurFlag"] = out.get("OurFlag", "").fillna("")
    out["BrkFlag"] = out.get("BrkFlag", "").fillna("")
    out["MatchID"] = out.get("MatchID", "").fillna("")
    out["Symbol"] = out.get("Symbol", "").fillna("").astype(str)
    out["Description"] = out.get("Description", "").fillna("").astype(str)
    return out


def _is_div_ref(sym: str) -> bool:
    return isinstance(sym, str) and sym.strip().startswith("89")


def _is_div_at(desc: str) -> bool:
    return isinstance(desc, str) and bool(_DIV_WORDS.search(desc))


def _tag_match(out: pd.DataFrame, rows: list[int], tag: str):
    for r in rows:
        if out.at[r, "__Side"] == "our":
            out.at[r, "OurFlag"] = "MATCHED"
        elif out.at[r, "__Side"] == "brk":
            out.at[r, "BrkFlag"] = "MATCHED"
        out.at[r, "MatchID"] = tag


# ----- step A: handle dividend references ("89...") -----
def _match_dividends(out: pd.DataFrame, tol: float) -> pd.DataFrame:
    res = out.copy()
    base = (res["OurFlag"] == "") & (res["BrkFlag"] == "")

    # counter for MATCH # tags
    counter = 1 + res["MatchID"].astype(str).str.contains("#", na=False).sum()

    # group all broker rows by the "89..." reference
    div_groups = {}
    for i, r in res.loc[base & (res["__Side"] == "brk")].iterrows():
        sym = r["Symbol"].strip()
        if _is_div_ref(sym):
            div_groups.setdefault(sym, []).append(i)

    # find candidate AT dividend rows (one per payout)
    at_div_idxs = [
        i
        for i, r in res.loc[base & (res["__Side"] == "our")].iterrows()
        if _is_div_at(r["Description"])
    ]

    for ref, brk_rows in div_groups.items():
        brk_sum = _r_round2(sum(res.at[i, "Diff"] for i in brk_rows))  # broker-only rows: Diff == Broker amount
        # find one AT dividend row whose amount equals -brk_sum (within tol)
        best = None
        best_gap = None
        for i in at_div_idxs:
            if res.at[i, "OurFlag"] or res.at[i, "BrkFlag"]:
                continue
            gap = _r_round2(res.at[i, "Diff"] + brk_sum)  # AT-only rows: Diff == -AT amount
            if abs(gap) <= tol and (best is None or abs(gap) < best_gap):
                best = i
                best_gap = abs(gap)
        if best is not None:
            tag = f"MATCH #{int(counter):05d}"
            _tag_match(res, [best] + brk_rows, tag)
            counter += 1

    return res


# ----- step B: amount-to-amount for everything else (strict 1-to-1 on absolute amount) -----
def _match_amount_to_amount(res: pd.DataFrame, tol: float) -> pd.DataFrame:
    out = res.copy()
    base = (out["OurFlag"] == "") & (out["BrkFlag"] == "")

    ours = [i for i, r in out.loc[base & (out["__Side"] == "our")].iterrows()]
    brks = [i for i, r in out.loc[base & (out["__Side"] == "brk")].iterrows()]

    # build maps by absolute amount rounded to 2dp
    def key(v):
        return _r_round2(abs(float(v)))

    ours_by_amt = {}
    for i in ours:
        a = key(out.at[i, "Diff"])  # our Diff == -AT amount
        ours_by_amt.setdefault(a, []).append(i)

    brk_by_amt = {}
    for j in brks:
        b = key(out.at[j, "Diff"])  # brk Diff == Broker amount
        brk_by_amt.setdefault(b, []).append(j)

    counter = 1 + out["MatchID"].astype(str).str.contains("#", na=False).sum()

    # match greedily by equal absolute amount
    for amt, o_idxs in list(ours_by_amt.items()):
        b_idxs = brk_by_amt.get(amt, [])
        while o_idxs and b_idxs:
            oi = o_idxs.pop()
            bj = b_idxs.pop()
            # sign should be opposite; enforce tolerance
            if abs(_r_round2(out.at[oi, "Diff"] + out.at[bj, "Diff"])) <= tol:
                tag = f"MATCH #{int(counter):05d}"
                _tag_match(out, [oi, bj], tag)
                counter += 1
            # if fails tolerance, leave them unmatched

    return out


def auto_match(rec: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """
    Riyadh Capital matching:
      1) For broker refs starting with '89': sum all broker rows per ref and match to one AT dividend row.
      2) Everything else: strict amount-to-amount 1↔1 by absolute value (within tol).
    Dates are not used in the matching logic.
    """
    if rec is None or rec.empty:
        return rec
    out = _r_prepare(rec)

    # A) dividends by "89..." reference
    out = _match_dividends(out, tol)

    # B) amount-to-amount for remaining rows
    out = _match_amount_to_amount(out, tol)

    return out


# ---- minimal copies of finalize helpers (same as in scb.py) ----
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

    rec = rec.sort_values(["Date", "Symbol", "Description"], kind="mergesort").reset_index(
        drop=True
    )
    rec["RowID"] = rec.index.astype(int) + 1
    rec["OurFlag"] = ""
    rec["BrkFlag"] = ""
    rec["MatchID"] = ""
    return rec


def _append_at(
    rec: pd.DataFrame, at_df: pd.DataFrame, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK
) -> pd.DataFrame:
    if at_df is None or at_df.empty:
        return rec
    at_part = at_df[[COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]]
    base = rec.drop(columns=["RowID", "OurFlag", "BrkFlag", "MatchID"], errors="ignore")
    merged = pd.concat([base, at_part], ignore_index=True)
    return _finalize(merged)
