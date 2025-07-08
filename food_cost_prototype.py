"""Food Cost Prototype – with Multi-Task AI Assistant
====================================================
Adds recipe creation, ingredient management, shopping list generation, and voice/text commands.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Streamlit & optional dependencies
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    def _noop(*args, **kwargs): return None
    class _DummyStreamlit:
        def __init__(self): self.sidebar=self; self.session_state={}
        def __getattr__(self, name): return _noop
        def __call__(self,*a,**k): return None
    st=_DummyStreamlit(); HAS_STREAMLIT=False

# OpenAI & Whisper API
try:
    import openai
    HAS_OPENAI=True
    # Load API key from environment or Streamlit secrets
    openai.api_key = os.getenv("OPENAI_API_KEY") or (
        st.secrets.get("OPENAI_API_KEY") if HAS_STREAMLIT and "OPENAI_API_KEY" in getattr(st, 'secrets', {}) else None
    )
except ModuleNotFoundError:
    openai=None; HAS_OPENAI=False

# Config
try:
    BASE_DIR=Path(__file__).parent
except NameError:
    BASE_DIR=Path.cwd()
DATA_DIR=BASE_DIR/"data"; DATA_DIR.mkdir(exist_ok=True)

# Table definitions
TABLE_SPECS: Dict[str, Dict[str, List[str]]] = {
    "ingredients": {"columns":[
        "item_id","name","purchase_unit","purchase_qty","purchase_price",
        "yield_percent","vendor","par_level_grams","lead_time_days",
        "last_updated","cost_per_gram","net_cost_per_gram"
    ]},
    "recipes": {"columns":["recipe_id","name","servings","cost_per_serving"]},
    "recipe_ingredients": {"columns":["recipe_id","ingredient_id","qty","unit","qty_grams","cost"]},
    "inventory_txn": {"columns":["txn_id","date","ingredient_id","qty_grams_change","reason","note"]},
    "labor_shift": {"columns":["shift_id","emp_email","start_time","end_time","hours","labor_cost"]},
}
UNIT_TO_GRAMS = {"lb":453.592,"oz":28.3495,"kg":1000,"g":1,"gal":3785.41,"ml":1,"l":1000}

# Data helpers

def _table_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def load_table(name: str) -> pd.DataFrame:
    cols = TABLE_SPECS[name]["columns"]
    path = _table_path(name)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in {"name","vendor","purchase_unit"} else 0
    df = df[cols]
    df.to_csv(path, index=False)
    return df


def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_table_path(name), index=False)


def calculate_cost_columns(row: pd.Series) -> pd.Series:
    unit = row.get("purchase_unit", "")
    qty = float(row.get("purchase_qty", 0) or 0)
    price = float(row.get("purchase_price", 0) or 0)
    yp = float(row.get("yield_percent", 1) or 1)
    if yp > 1: yp /= 100.0
    grams = qty * UNIT_TO_GRAMS.get(unit, 1)
    cpg = price / grams if grams else 0.0
    ncpg = cpg / yp if yp else 0.0
    row["cost_per_gram"] = round(cpg, 4)
    row["net_cost_per_gram"] = round(ncpg, 4)
    row["last_updated"] = datetime.utcnow().isoformat()
    return row


def current_stock() -> pd.DataFrame:
    inv = load_table("inventory_txn")
    ing = load_table("ingredients")
    stock = inv.groupby("ingredient_id")["qty_grams_change"].sum().reset_index()
    stock.rename(columns={"ingredient_id":"item_id","qty_grams_change":"on_hand_grams"}, inplace=True)
    return ing[["item_id","name","par_level_grams"]].merge(stock, on="item_id", how="left").fillna({"on_hand_grams":0})

# AI parser & multitask handler

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    # 1) Add ingredients from recipe to ingredients list
    if "ingredients" in t and "recipe" in t and re.search(r"add ingredients.*recipe", t):
        # extract recipe name
        m = re.search(r'recipe\s+"([^"]+)"', text, re.I)
        name = m.group(1) if m else re.search(r'recipe\s+(\w[\w\s]+)', text, re.I).group(1)
        # find recipe_id
        rdf = st.session_state["tables"]["recipes"]
        row = rdf[rdf['name'].str.lower() == name.strip().lower()]
        if row.empty:
            return f"❌ Recipe '{name}' not found."
        rid = row.iloc[0]['recipe_id']
        # get recipe ingredients
        ri_df = st.session_state["tables"]["recipe_ingredients"]
        items = ri_df[ri_df['recipe_id'] == rid]
        ing_df = st.session_state["tables"]["ingredients"] if "ingredients" in st.session_state.get("tables",{}) else load_table("ingredients")
        added = 0
        for _, rec in items.iterrows():
            # determine name and grams
            ingr = rec['ingredient_id']
            qty_grams = rec['qty_grams']
            # lookup name
            ing_name = ing_df[ing_df['item_id'] == ingr]['name'].iloc[0] if ingr else rec['ingredient_id']
            ex = ing_df[ing_df['name'].str.lower() == ing_name.strip().lower()]
            if ex.empty:
                new = {
                    'item_id': f'NEW{len(ing_df)+1}',
                    'name': ing_name.title(),
                    'purchase_unit': '',
                    'purchase_qty': qty_grams,
                    'purchase_price': 0,
                    'yield_percent': 100,
                    'vendor': '',
                    'par_level_grams': 0,
                    'lead_time_days': 0
                }
                ing_df = pd.concat([ing_df, pd.DataFrame([new])], ignore_index=True)
                added += 1
            else:
                idx = ex.index[0]
                if not ex.loc[idx,'purchase_unit']:
                    ing_df.loc[idx,'purchase_unit'] = 'g'
                if not ex.loc[idx,'purchase_qty']:
                    ing_df.loc[idx,'purchase_qty'] = qty_grams
                if not ex.loc[idx,'yield_percent']:
                    ing_df.loc[idx,'yield_percent'] = 100
                added += 1
        st.session_state['tables']['ingredients'] = ing_df
        save_table('ingredients', ing_df)
        return f"✅ Added {added} ingredients from recipe '{name}'. Please fill in missing purchase units/prices in Ingredients."
    # 2) Generate shopping list of low-stock items
    if "shopping list" in t:
        stock = current_stock()
        low = stock[stock['on_hand_grams'] < stock['par_level_grams']]
        if low.empty:
            return "🛒 Shopping list is empty: no items below par level."
        items = low['name'].tolist()
        return "🛒 Shopping list: " + ", ".join(items)
    # 3) Recipe creation (existing logic)
    if "recipe" in t:
        # ... (existing recipe creation code) ...
        pass
    # 4) Stock check
    if t.startswith("how many") and "left" in t:
        nm = t.replace("how many", "").replace("left", "").strip()
        row = current_stock()[lambda df: df['name'].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # 5) Add purchase
    m3 = re.match(r"add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)", t)
    if m3 and HAS_STREAMLIT:
        qty, unit, nm, pr = m3.groups()
        qty, pr = float(qty.replace(',', '.')), float(pr.replace(',', '.'))
        df = st.session_state['tables']['ingredients']
        ex = df[df['name'].str.lower() == nm.strip().lower()]
        if ex.empty:
            new = {'item_id': f'NEW{len(df)+1}', 'name': nm.title(), 'purchase_unit': unit, 'purchase_qty': qty, 'purchase_price': pr, 'yield_percent': 100}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        else:
            idx = ex.index[0]
            df.loc[idx, ['purchase_unit','purchase_qty','purchase_price','yield_percent']] = [unit, qty, pr, 100]
        df = df.apply(calculate_cost_columns, axis=1)
        st.session_state['tables']['ingredients'] = df
        save_table('ingredients', df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    # 6) GPT fallback
    if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
        resp = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[{'role':'system','content':'You are a food-truck cost app assistant.'},{'role':'user','content':text}]
        )
        return resp.choices[0].message['content'].strip()
    return "🤔 Sorry, I couldn’t parse that."

# Transcription helper
def transcribe_audio(audio_file):
    if HAS_OPENAI and audio_file:
        return openai.Audio.transcribe('whisper-1', audio_file)
    return ''

# Streamlit UI remains unchanged
# ...
