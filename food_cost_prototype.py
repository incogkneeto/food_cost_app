"""Food Cost Prototype – works with or without Streamlit
====================================================
This script powers a food-cost app.  When the **streamlit** package is
installed, it spins up a full interactive UI.  In head-less or sandboxed
contexts (like many test runners), it swaps Streamlit for a tiny *dummy*
namespace so that the data-layer logic can still be imported and tested
without raising errors.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd

#-----------------------------------------------------------------------------#
# Attempt to import Streamlit; fallback to dummy for non-UI contexts         #
#-----------------------------------------------------------------------------#
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    def _noop(*args, **kwargs):
        return None
    class _DummyStreamlit:
        def __init__(self):
            self.sidebar = self
            self.session_state = {}
        def __getattr__(self, name):
            return _noop
        def __call__(self, *args, **kwargs):
            return None
    st = _DummyStreamlit()  # type: ignore
    HAS_STREAMLIT = False

#-----------------------------------------------------------------------------#
# Attempt to import OpenAI; fallback if not available                       #
#-----------------------------------------------------------------------------#
try:
    import openai
    HAS_OPENAI = True
except ModuleNotFoundError:
    openai = None  # type: ignore
    HAS_OPENAI = False

#-----------------------------------------------------------------------------#
# Configuration constants                                                   #
#-----------------------------------------------------------------------------#
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TABLE_SPECS: Dict[str, Dict[str, List[str]]] = {
    "ingredients": {"columns": [
        "item_id","name","purchase_unit","purchase_qty","purchase_price",
        "yield_percent","vendor","par_level_grams","lead_time_days",
        "last_updated","cost_per_gram","net_cost_per_gram"
    ]},
    "recipes": {"columns": ["recipe_id","name","servings","cost_per_serving"]},
    "recipe_ingredients": {"columns": ["recipe_id","ingredient_id","qty","unit","qty_grams","cost"]},
    "inventory_txn": {"columns": ["txn_id","date","ingredient_id","qty_grams_change","reason","note"]},
    "labor_shift": {"columns": ["shift_id","emp_email","start_time","end_time","hours","labor_cost"]},
}
UNIT_TO_GRAMS = {"lb":453.592,"oz":28.3495,"kg":1000,"g":1,"gal":3785.41,"ml":1,"l":1000}

#-----------------------------------------------------------------------------#
# Data helpers                                                               #
#-----------------------------------------------------------------------------#

def _table_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def load_table(name: str) -> pd.DataFrame:
    """Load a table from disk, creating it with the latest schema if missing."""
    cols = TABLE_SPECS[name]["columns"]
    path = _table_path(name)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    # ensure schema
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in {"name","vendor","purchase_unit"} else 0
    df = df[cols]
    df.to_csv(path, index=False)
    return df


def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_table_path(name), index=False)


def calculate_cost_columns(row: pd.Series) -> pd.Series:
    """Compute cost_per_gram & net_cost_per_gram for an ingredient row."""
    unit = row.get("purchase_unit", "")
    qty = float(row.get("purchase_qty", 0) or 0)
    price = float(row.get("purchase_price", 0) or 0)
    yp = float(row.get("yield_percent", 1) or 1)
    if yp > 1:
        yp /= 100.0
    grams = qty * UNIT_TO_GRAMS.get(unit, 1)
    cpg = price / grams if grams else 0.0
    ncpg = cpg / yp if yp else 0.0
    row["cost_per_gram"] = round(cpg, 4)
    row["net_cost_per_gram"] = round(ncpg, 4)
    row["last_updated"] = datetime.utcnow().isoformat()
    return row


def current_stock() -> pd.DataFrame:
    """Return on-hand grams per ingredient."""
    inv = (st.session_state.get("tables", {}).get("inventory_txn") if HAS_STREAMLIT and "tables" in st.session_state else load_table("inventory_txn"))
    ing = (st.session_state.get("tables", {}).get("ingredients") if HAS_STREAMLIT and "tables" in st.session_state else load_table("ingredients"))
    stock = inv.groupby("ingredient_id")["qty_grams_change"].sum().reset_index()
    stock.rename(columns={"ingredient_id":"item_id","qty_grams_change":"on_hand_grams"}, inplace=True)
    return ing[["item_id","name"]].merge(stock, on="item_id", how="left").fillna({"on_hand_grams":0})

#-----------------------------------------------------------------------------#
# AI COMMAND PARSER & INSIGHTS                                                 #
#-----------------------------------------------------------------------------#

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    if t.startswith("how many") and "left" in t:
        nm = t.replace("how many", "").replace("left", "").strip()
        row = current_stock()[lambda df: df["name"].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    m = re.match(r"add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)", t)
    if m and HAS_STREAMLIT:
        qty, unit, nm, pr = m.groups()
        qty, pr = float(qty.replace(",", ".")), float(pr.replace(",", "."))
        df = st.session_state["tables"]["ingredients"]
        r = df[df["name"].str.lower() == nm]
        if r.empty:
            new = {"item_id": f"NEW{len(df)+1}", "name": nm.title().strip(),
                   "purchase_unit": unit, "purchase_qty": qty, "purchase_price": pr, "yield_percent": 1.0}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        else:
            idx = r.index[0]
            df.loc[idx, ["purchase_unit","purchase_qty","purchase_price"]] = [unit, qty, pr]
        df = df.apply(calculate_cost_columns, axis=1)
        st.session_state["tables"]["ingredients"] = df
        save_table("ingredients", df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are an assistant for a food-truck cost app. Return concise answers."},
                {"role":"user","content": text},
            ],
        )
        return resp.choices[0].message["content"].strip()
    return "🤔 Sorry, I couldn’t parse that."

def get_ai_insights(df: pd.DataFrame) -> str:
    if not (HAS_OPENAI and os.getenv("OPENAI_API_KEY")):
        return "AI not configured – set OPENAI_API_KEY to enable insights."
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a food-cost analyst. Suggest cost-control tips."},
            {"role":"user","content": df.to_csv(index=False)},
        ],
    )
    return resp.choices[0].message["content"].strip()

#-----------------------------------------------------------------------------#
# Basic self-tests                                                             #
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    # Test data-layer functions
    ing = load_table("ingredients")
    assert isinstance(ing, pd.DataFrame), "load_table should return DataFrame"
    # Test cost calculation logic
    test_row = pd.Series({"purchase_unit":"g","purchase_qty":100,"purchase_price":2,"yield_percent":100})
    calc = calculate_cost_columns(test_row.copy())
    assert calc["cost_per_gram"] == 0.02, f"Expected 0.02, got {calc['cost_per_gram']}"
    print("All basic tests passed.")
