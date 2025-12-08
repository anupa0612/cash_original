# brokers/scb.py
import io
import re
import pandas as pd

# ===========================
# ------- Cleaning ----------
# ===========================
_NEG_PARENS_RX = re.compile(r"\(([^)]+)\)")  # "(1,234.56)" -> "-1234.56"

def _clean_num_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([0.0] * 0, dtype=float)
    s = s.astype(str).str.strip()
    s = s.where(s.ne(""), "0")
    s = s.str.replace(",", "", regex=False)
    s = s.apply(lambda x: "-" + _NEG_PARENS_RX.sub(r"\1", x) if _NEG_PARENS_RX.fullmatch(x) else x)
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

def _amount_from_with_dep(withdrawal: pd.Series, deposit: pd.Series) -> pd.Series:
    w = _clean_num_series(withdrawal)
    d = _clean_num_series(deposit)
    return (d - w).astype(float)

def clean_broker(file_storage, start_date_str: str, end_date_str: str, account_value: str = None) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date_str, errors="coerce")
    end_dt = pd.to_datetime(end_date_str, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("Invalid date range. Please check From / To dates.")

    raw = file_storage.read()
    file_storage.stream.seek(0)
    try:
        df = pd.read_excel(io.BytesIO(raw), dtype=str, engine=None)
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if df is None or df.empty:
        return pd.DataFrame(columns=["SettleDate", "ReportDate", "Symbol", "Desc", "Amount"])

    cmap = {c.lower().strip(): c for c in df.columns}
    need = ["date", "description", "withdrawal", "deposit", "account number"]
    miss = [k for k in need if k not in cmap]
    if miss:
        raise ValueError("SCB file missing required columns: " + ", ".join(miss))

    c_date = cmap["date"]
    c_desc = cmap["description"]
    c_w = cmap["withdrawal"]
    c_d = cmap["deposit"]
    c_acct = cmap["account number"]

    if account_value:
        df = df.loc[df[c_acct].astype(str).str.strip() == str(account_value).strip()]
        if df.empty:
            raise ValueError(f"No rows found for Account Number '{account_value}' in the uploaded file.")

    uniq = df[c_acct].astype(str).str.strip().unique()
    if account_value is None and len([u for u in uniq if u != ""]) > 1:
        raise ValueError(
            "Multiple Account Numbers detected in the SCB file. "
            "Please choose an Account in the UI and ensure the backend passes it through."
        )

    settle = pd.to_datetime(df[c_date].astype(str).str.strip(), format="%d-%b-%y", errors="coerce")
    amt = _amount_from_with_dep(df[c_w], df[c_d])

    out = pd.DataFrame({
        "SettleDate": settle,
        "ReportDate": settle,
        "Symbol": "",
        "Desc": df[c_desc].astype(str).fillna(""),
        "Amount": amt,
    })

    # EXCLUDE balance lines (case-insensitive), incl. 'balance b/f'
    excl = out["Desc"].str.contains(r"(?i)\b(closing balance|balance brought forward|balance b/?f)\b", na=False)
    out = out.loc[~excl]  # <-- FIX: use ~, not !
    out = out.dropna(subset=["SettleDate"]).copy()
    out = out.loc[out["Amount"].abs() > 1e-9]
    mask_range = (out["SettleDate"] >= start_dt) & (out["SettleDate"] <= end_dt)
    out = out.loc[mask_range].reset_index(drop=True)
    return out

# ===========================
# -------- Builder ----------
# ===========================
def build_rec(at_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    COL_DATE, COL_DESC, COL_AT, COL_BRK, COL_SYMBOL = "Date", "Description", "AT", "Broker", "Symbol"

    if broker_df is not None and not broker_df.empty:
        brk = pd.DataFrame({
            COL_DATE:   broker_df["SettleDate"],
            COL_SYMBOL: "",
            COL_DESC:   broker_df["Desc"].astype(str),
            COL_AT:     0.0,
            COL_BRK:    broker_df["Amount"].astype(float),
            "Comments": "",
        })
    else:
        # <<< typed empty DataFrame >>>
        brk = pd.DataFrame({
            COL_DATE:   pd.Series([], dtype="datetime64[ns]"),
            COL_SYMBOL: pd.Series([], dtype=str),
            COL_DESC:   pd.Series([], dtype=str),
            COL_AT:     pd.Series([], dtype="float64"),
            COL_BRK:    pd.Series([], dtype="float64"),
            "Comments": pd.Series([], dtype=str),
        })

    rec = _finalize(brk)
    rec = _append_at(rec, at_df, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK)

    # <<< belt-and-suspenders: enforce numeric >>> 
    rec[COL_AT]  = pd.to_numeric(rec[COL_AT],  errors="coerce").fillna(0.0).astype("float64")
    rec[COL_BRK] = pd.to_numeric(rec[COL_BRK], errors="coerce").fillna(0.0).astype("float64")
    return rec


# ===========================
# ---- SCB Auto-Matching ----
# ===========================
_SD_RX = re.compile(r"(?i)\bsettled\s+trades\s+sd\s+(\d{1,2}[A-Za-z]{3}\d{2,4})\b")

def _round2(x):
    return float(pd.Series([x]).round(2).iloc[0])

def _side(at_val, brk_val):
    a = _round2(abs(float(at_val))); b = _round2(abs(float(brk_val)))
    if a > 0 and b == 0: return "our"
    if b > 0 and a == 0: return "brk"
    if a == 0 and b == 0: return "empty"
    return "mixed"

def _extract_sd_date(desc: str):
    if not isinstance(desc, str):
        return ""
    m = _SD_RX.search(desc)
    if not m:
        return ""
    raw = m.group(1)
    dt = pd.to_datetime(raw, format="%d%b%Y", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(raw, format="%d%b%y", errors="coerce")
    return "" if pd.isna(dt) else dt.strftime("%Y-%m-%d")

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DateKey"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Diff"] = (out["Broker"] - out["AT"]).apply(_round2)
    out["__Side"] = out.apply(lambda r: _side(r["AT"], r["Broker"]), axis=1)
    out["OurFlag"] = out["OurFlag"].fillna("")
    out["BrkFlag"] = out["BrkFlag"].fillna("")
    out["MatchID"] = out["MatchID"].fillna("")
    out["_SDKey"] = out["Description"].apply(_extract_sd_date)
    return out

def _pair_two_row(df: pd.DataFrame, group_cols: list[str], tol: float) -> pd.DataFrame:
    out = df.copy()
    base = (out["OurFlag"] == "") & (out["BrkFlag"] == "") & (~out["Diff"].isna())
    if not base.any():
        return out
    counter = 1 + out["MatchID"].astype(str).str.contains("#", na=False).sum()
    for _, g in out.loc[base].groupby(group_cols, dropna=False):
        cand = [i for i in g.index if out.at[i, "__Side"] in ("our", "brk") and _round2(out.at[i, "Diff"]) != 0.0]
        if len(cand) < 2:
            continue
        cand.sort(key=lambda i: out.at[i, "Diff"])
        L, R = 0, len(cand) - 1
        used = set()
        while L < R:
            i, j = cand[L], cand[R]
            if i in used:
                L += 1; continue
            if j in used:
                R -= 1; continue
            s = _round2(float(out.at[i, "Diff"] + out.at[j, "Diff"]))
            if abs(s) <= tol and out.at[i, "__Side"] != out.at[j, "__Side"]:
                tag = f"MATCH #{counter:05d}"
                for r in (i, j):
                    if out.at[r, "__Side"] == "our": out.at[r, "OurFlag"] = "MATCHED"
                    else: out.at[r, "BrkFlag"] = "MATCHED"
                    out.at[r, "MatchID"] = tag
                used.update((i, j))
                counter += 1
                L += 1; R -= 1
            elif s < 0:
                L += 1
            else:
                R -= 1
    return out

def _subset_sum(values, rows, target, tol=0.01, max_n=18):
    vals = values[:max_n]; idxs = rows[:max_n]
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

def _pair_one_to_many(df: pd.DataFrame, group_cols: list[str], tol: float, max_n: int = 25) -> pd.DataFrame:
    out = df.copy()
    base = (out["OurFlag"] == "") & (out["BrkFlag"] == "") & (~out["Diff"].isna())
    if not base.any():
        return out
    counter = 1 + out["MatchID"].astype(str).str.contains("#", na=False).sum()
    for _, g in out.loc[base].groupby(group_cols, dropna=False):
        remaining = [i for i in g.index if out.at[i, "__Side"] in ("our", "brk") and _round2(out.at[i, "Diff"]) != 0.0]
        remaining.sort(key=lambda i: abs(out.at[i, "Diff"]), reverse=True)
        used = set()
        for anchor in list(remaining):
            if anchor in used: continue
            if out.at[anchor, "OurFlag"] or out.at[anchor, "BrkFlag"]:
                used.add(anchor); continue
            a_side = out.at[anchor, "__Side"]
            a_diff = _round2(float(out.at[anchor, "Diff"]))
            if a_side not in ("our", "brk") or a_diff == 0.0:
                used.add(anchor); continue
            target = _round2(-a_diff)
            if target == 0.0:
                used.add(anchor); continue
            pool_idxs, pool_vals = [], []
            for j in remaining:
                if j == anchor or j in used: continue
                if out.at[j, "__Side"] == a_side: continue
                d = _round2(float(out.at[j, "Diff"]))
                if d == 0.0: continue
                if (target > 0 and d <= 0) or (target < 0 and d >= 0): continue
                pool_idxs.append(j); pool_vals.append(abs(d))
            if not pool_idxs:
                used.add(anchor); continue
            order = sorted(range(len(pool_vals)), key=lambda k: -pool_vals[k])
            vals_sorted = [pool_vals[k] for k in order]
            rows_sorted = [pool_idxs[k] for k in order]
            chosen = _subset_sum(vals_sorted, rows_sorted, abs(target), tol, max_n=max_n)
            if not chosen:
                used.add(anchor); continue
            tag = f"MATCH #{counter:05d}"
            if a_side == "our": out.at[anchor, "OurFlag"] = "MATCHED"
            else: out.at[anchor, "BrkFlag"] = "MATCHED"
            out.at[anchor, "MatchID"] = tag
            for r in chosen:
                if out.at[r, "__Side"] == "our": out.at[r, "OurFlag"] = "MATCHED"
                else: out.at[r, "BrkFlag"] = "MATCHED"
                out.at[r, "MatchID"] = tag
                used.add(r)
            used.add(anchor)
            counter += 1
    return out

def auto_match(rec: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    if rec is None or rec.empty:
        return rec
    out = _prepare(rec)
    if out["_SDKey"].ne("").any():
        out = _pair_two_row(out, ["_SDKey"], tol)
        out = _pair_one_to_many(out, ["_SDKey"], tol, max_n=25)
    out = _pair_two_row(out, ["DateKey"], tol)
    out = _pair_one_to_many(out, ["DateKey"], tol, max_n=25)
    return out

# ===========================
# ------- Finalize ----------
# ===========================
def _finalize(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        base = pd.DataFrame({
            "Date":        pd.Series([], dtype="datetime64[ns]"),
            "Symbol":      pd.Series([], dtype=str),
            "Description": pd.Series([], dtype=str),
            "AT":          pd.Series([], dtype="float64"),
            "Broker":      pd.Series([], dtype="float64"),
            "Comments":    pd.Series([], dtype=str),
        })
        base["RowID"]   = pd.Series([], dtype="int64")
        base["OurFlag"] = pd.Series([], dtype=str)
        base["BrkFlag"] = pd.Series([], dtype=str)
        base["MatchID"] = pd.Series([], dtype=str)
        return base

    rec = rec.sort_values(["Date", "Symbol", "Description"], kind="mergesort").reset_index(drop=True)
    rec["RowID"]   = rec.index.astype(int) + 1
    rec["OurFlag"] = ""
    rec["BrkFlag"] = ""
    rec["MatchID"] = ""
    return rec


def _append_at(rec: pd.DataFrame, at_df: pd.DataFrame, COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK) -> pd.DataFrame:
    """
    SCB variant: append AT rows but force Symbol = "" for all AT lines.
    """
    if at_df is None or at_df.empty:
        return rec

    # take needed columns safely
    cols = [COL_DATE, COL_SYMBOL, COL_DESC, COL_AT, COL_BRK, "Comments"]
    missing = [c for c in cols if c not in at_df.columns]
    if missing:
        # create any missing columns to avoid KeyErrors
        at_df = at_df.copy()
        for c in missing:
            at_df[c] = "" if c in (COL_SYMBOL, COL_DESC, "Comments") else 0.0

    at_part = at_df[cols].copy()

    # **IMPORTANT**: make Symbol empty for AT lines in SCB flow
    at_part[COL_SYMBOL] = ""   # <- this makes the Symbol column empty

    base = rec.drop(columns=["RowID", "OurFlag", "BrkFlag", "MatchID"], errors="ignore")
    merged = pd.concat([base, at_part], ignore_index=True)
    return _finalize(merged)

