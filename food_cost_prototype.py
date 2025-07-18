"""Food Cost Prototype – with Multi-Task AI Assistant
====================================================
Streamlit app for managing food costs: ingredients, recipes, inventory, labor, AI assistant with voice/text commands.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Streamlit & fallback
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    def _noop(*args, **kwargs): return None
    class _DummyStreamlit:
        def __init__(self):
            self.sidebar = self
            self.session_state = {}
            self.secrets = {}
        def __getattr__(self, name): return _noop
        def __call__(self, *args, **kwargs): return None
    st = _DummyStreamlit()
    HAS_STREAMLIT = False

# OpenAI & Whisper
try:
    import openai
    HAS_OPENAI = True
    openai.api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if HAS_STREAMLIT else None)
except Exception:
    openai = None
    HAS_OPENAI = False

# Config dirs
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Table schemas
TABLE_SPECS: Dict[str, List[str]] = {
    'ingredients': [
        'item_id','name','purchase_unit','purchase_qty','purchase_price',
        'yield_percent','vendor','par_level_grams','lead_time_days',
        'last_updated','cost_per_gram','net_cost_per_gram'
    ],
    'recipes': ['recipe_id','name','servings','cost_per_serving'],
    'recipe_ingredients': ['recipe_id','ingredient_id','qty','unit','qty_grams','cost'],
    'inventory_txn': ['txn_id','date','ingredient_id','qty_grams_change','reason','note'],
    'labor_shift': ['shift_id','emp_email','start_time','end_time','hours','labor_cost'],
}
UNIT_TO_GRAMS = {'lb':453.592,'oz':28.3495,'kg':1000,'g':1,'gal':3785.41,'ml':1,'l':1000}

# Data helpers

def _table_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"

def load_table(name: str) -> pd.DataFrame:
    cols = TABLE_SPECS[name]
    path = _table_path(name)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = '' if c in ['name','vendor','purchase_unit'] else 0
    return df[cols]

def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_table_path(name), index=False)

def calculate_cost_columns(row: pd.Series) -> pd.Series:
    unit = row.get('purchase_unit', '')
    qty = float(row.get('purchase_qty', 0) or 0)
    price = float(row.get('purchase_price', 0) or 0)
    yp = float(row.get('yield_percent', 1) or 1)
    if yp > 1:
        yp /= 100.0
    grams = qty * UNIT_TO_GRAMS.get(unit, 1)
    cpg = price / grams if grams else 0.0
    nc = cpg / yp if yp else 0.0
    row['cost_per_gram'] = round(cpg, 4)
    row['net_cost_per_gram'] = round(nc, 4)
    row['last_updated'] = datetime.utcnow().isoformat()
    return row

def current_stock() -> pd.DataFrame:
    inv = load_table('inventory_txn')
    ing = load_table('ingredients')
    stock = inv.groupby('ingredient_id')['qty_grams_change'].sum().reset_index()
    stock.rename(columns={'ingredient_id':'item_id','qty_grams_change':'on_hand_grams'}, inplace=True)
    df = ing.merge(stock, on='item_id', how='left').fillna({'on_hand_grams':0})
    return df[['item_id','name','par_level_grams','on_hand_grams']]

# AI parse and multitask

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    # 1) Add ingredients from any recipe
    m = re.search(r'add(?: the)? (?:ingredients from|recipe)\s+"?(.+?)"?(?: to ingredients)?', text, re.I)
    if m:
        name = m.group(1).strip()
        recipes = st.session_state.get('tables', {}).get('recipes', load_table('recipes'))
        row = recipes[recipes['name'].str.lower() == name.lower()]
        if row.empty:
            return f"❌ Recipe '{name}' not found."
        rid = row.iloc[0]['recipe_id']
        ri = st.session_state.get('tables', {}).get('recipe_ingredients', load_table('recipe_ingredients'))
        items = ri[ri['recipe_id'] == rid]
        ing_df = st.session_state.get('tables', {}).get('ingredients', load_table('ingredients'))
        added = 0
        for _, rec in items.iterrows():
            item_id = rec['ingredient_id']
            grams = rec['qty_grams']
            names = ing_df.loc[ing_df['item_id'] == item_id, 'name'].tolist()
            ing_name = names[0] if names else ''
            ex = ing_df[ing_df['name'].str.lower() == ing_name.lower()]
            if ex.empty:
                new = {
                    'item_id': f'NEW{len(ing_df)+1}', 'name': ing_name.title(),
                    'purchase_unit': '',      'purchase_qty': grams,
                    'purchase_price': 0,       'yield_percent': 100,
                    'vendor': '',             'par_level_grams': 0,
                    'lead_time_days': 0
                }
                ing_df = pd.concat([ing_df, pd.DataFrame([new])], ignore_index=True)
            else:
                idx = ex.index[0]
                if not ing_df.at[idx,'purchase_unit']:
                    ing_df.at[idx,'purchase_unit'] = 'g'
                if not ing_df.at[idx,'purchase_qty']:
                    ing_df.at[idx,'purchase_qty'] = grams
                if not ing_df.at[idx,'yield_percent']:
                    ing_df.at[idx,'yield_percent'] = 100
            added += 1
        st.session_state['tables']['ingredients'] = ing_df
        save_table('ingredients', ing_df)
        return f"✅ Added {added} ingredients from recipe '{name}' to Ingredients list."

    # 2) Add purchase – two patterns:
    if HAS_STREAMLIT:
        # a) qty-first: “add 25 lb tomatoes @ 18.99”
        m_qty = re.search(r'add.*?(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w ]+?)\s*@\s*\$?(\d+[.,]?\d*)', t)
        if m_qty:
            qty, unit, nm, pr = m_qty.groups()
            qty, pr = float(qty.replace(',','.')), float(pr.replace(',','.'))
        else:
            # b) name-first: “add tomatoes to my ingredients 25 lb @ 18.99”
            m_name = re.search(
                r'add\s+([\w ]+?)\s+(?:to (?:my )?ingredients)?\s*(\d+[.,]?\d*)\s*'
                r'(lb|oz|kg|g)\s*@\s*\$?(\d+[.,]?\d*)',
                t
            )
            if not m_name:
                m_qty = None; m_name = None
            else:
                nm, qty, unit, pr = m_name.groups()
                qty, pr = float(qty.replace(',','.')), float(pr.replace(',','.'))

        # if either matched:
        if (HAS_STREAMLIT and (m_qty or m_name)):
            ing_df = st.session_state['tables']['ingredients']
            ex = ing_df[ing_df['name'].str.lower() == nm.strip().lower()]
            if ex.empty:
                new = {
                    'item_id': f'NEW{len(ing_df)+1}', 'name': nm.title(),
                    'purchase_unit': unit,       'purchase_qty': qty,
                    'purchase_price': pr,        'yield_percent': 100
                }
                ing_df = pd.concat([ing_df, pd.DataFrame([new])], ignore_index=True)
            else:
                idx = ex.index[0]
                ing_df.loc[idx, ['purchase_unit','purchase_qty','purchase_price','yield_percent']] = [unit, qty, pr, 100]
            ing_df = ing_df.apply(calculate_cost_columns, axis=1)
            st.session_state['tables']['ingredients'] = ing_df
            save_table('ingredients', ing_df)
            return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"

    # 3) Shopping list
    if 'shopping list' in t:
        stock = current_stock()
        low = stock[stock['on_hand_grams'] < stock['par_level_grams']]
        return (
            "🛒 Shopping list: " + ", ".join(low['name'].tolist())
            if not low.empty else "🛒 No items below par level."
        )

    # 4) Stock check
    if t.startswith('how many') and 'left' in t:
        nm = t.replace('how many','').replace('left','').strip()
        row = current_stock()[lambda df: df['name'].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"

    # 5) GPT fallback
    if HAS_OPENAI and openai.api_key:
        resp = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[
                {'role':'system','content':'You are a food-truck cost app assistant.'},
                {'role':'user','content':text}
            ]
        )
        return resp.choices[0].message['content'].strip()

    return "🤔 Sorry, I couldn’t parse that."


# Transcribe helper

def transcribe_audio(audio_file):
    return openai.Audio.transcribe('whisper-1', audio_file) if HAS_OPENAI and audio_file else ''

# Streamlit UI
# ... rest of UI unchanged ...
