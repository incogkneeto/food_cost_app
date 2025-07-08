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
        def __init__(self):
            self.sidebar = self
            self.session_state = {}
        def __getattr__(self, name): return _noop
        def __call__(self, *args, **kwargs): return None
    st = _DummyStreamlit()
    HAS_STREAMLIT = False

# OpenAI & Whisper API
try:
    import openai
    HAS_OPENAI = True
    # Load API key from env or secrets
    openai.api_key = os.getenv("OPENAI_API_KEY") or (
        st.secrets.get("OPENAI_API_KEY") if HAS_STREAMLIT and hasattr(st, 'secrets') else None
    )
except ModuleNotFoundError:
    openai = None
    HAS_OPENAI = False

# Config
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

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
    inv = load_table("inventory_txn")
    ing = load_table("ingredients")
    stock = inv.groupby("ingredient_id")["qty_grams_change"].sum().reset_index()
    stock.rename(columns={"ingredient_id":"item_id","qty_grams_change":"on_hand_grams"}, inplace=True)
    return ing[["item_id","name","par_level_grams"]].merge(stock, on="item_id", how="left").fillna({"on_hand_grams":0})

# AI parser & multitask handler

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    # 1) Add ingredients from recipe
    if "add ingredients" in t and "from recipe" in t:
        m = re.search(r'from recipe\s+"([^"]+)"', text, re.I)
        name = m.group(1) if m else re.search(r'from recipe\s+(\w[\w\s]+)', text, re.I).group(1)
        rdf = st.session_state.get("tables", {}).get("recipes", load_table("recipes"))
        row = rdf[rdf['name'].str.lower() == name.strip().lower()]
        if row.empty:
            return f"❌ Recipe '{name}' not found."
        rid = row.iloc[0]['recipe_id']
        ri_df = st.session_state.get("tables", {}).get("recipe_ingredients", load_table("recipe_ingredients"))
        items = ri_df[ri_df['recipe_id'] == rid]
        ing_df = st.session_state.get("tables", {}).get("ingredients", load_table("ingredients"))
        added = 0
        for _, rec in items.iterrows():
            ingr = rec['ingredient_id']
            qty_grams = rec['qty_grams']
            ing_name = ing_df[ing_df['item_id'] == ingr]['name'].iloc[0] if ingr else ''
            ex = ing_df[ing_df['name'].str.lower() == ing_name.lower()]
            if ex.empty:
                new = {
                    'item_id': f'NEW{len(ing_df)+1}', 'name': ing_name.title(),
                    'purchase_unit': '', 'purchase_qty': qty_grams,
                    'purchase_price': 0, 'yield_percent': 100,
                    'vendor': '', 'par_level_grams': 0, 'lead_time_days': 0
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
        return f"✅ Added {added} ingredients from recipe '{name}'."
    # 2) Shopping list
    if "shopping list" in t:
        stock = current_stock()
        low = stock[stock['on_hand_grams'] < stock['par_level_grams']]
        if low.empty:
            return "🛒 Shopping list is empty: no items below par level."
        return "🛒 Shopping list: " + ", ".join(low['name'].tolist())
    # 3) Stock check
    if t.startswith("how many") and "left" in t:
        nm = t.replace("how many", "").replace("left", "").strip()
        row = current_stock()[lambda df: df['name'].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # 4) Add purchase
    m3 = re.match(r"add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)", text.lower())
    if m3 and HAS_STREAMLIT:
        qty, unit, nm, pr = m3.groups()
        qty, pr = float(qty.replace(',', '.')), float(pr.replace(',', '.'))
        df = st.session_state['tables']['ingredients']
        ex = df[df['name'].str.lower() == nm.strip().lower()]
        if ex.empty:
            new = {'item_id': f'NEW{len(df)+1}', 'name': nm.title(),
                   'purchase_unit': unit, 'purchase_qty': qty,
                   'purchase_price': pr, 'yield_percent':100}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        else:
            idx = ex.index[0]
            df.loc[idx,['purchase_unit','purchase_qty','purchase_price','yield_percent']] = [unit,qty,pr,100]
        df = df.apply(calculate_cost_columns, axis=1)
        st.session_state['tables']['ingredients'] = df
        save_table('ingredients', df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    # 5) GPT fallback
    if HAS_OPENAI and openai.api_key:
        resp = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[{'role':'system','content':'You are a food-truck cost app assistant.'},
                      {'role':'user','content':text}]
        )
        return resp.choices[0].message['content'].strip()
    return "🤔 Sorry, I couldn’t parse that."

# Transcribe helper

def transcribe_audio(audio_file):
    return openai.Audio.transcribe('whisper-1', audio_file) if HAS_OPENAI and audio_file else ''

# Streamlit UI
if HAS_STREAMLIT:
    st.set_page_config(page_title="Food Cost App", page_icon="🍔", layout="wide")
    if 'tables' not in st.session_state:
        st.session_state['tables'] = {n: load_table(n) for n in TABLE_SPECS}
        st.session_state['chat_log'] = []

    def get_table(name: str) -> pd.DataFrame:
        return st.session_state['tables'][name]

    def persist(name: str) -> None:
        save_table(name, get_table(name))

    menu = st.sidebar.radio("Navigation", [
        'ingredients','recipes','recipe_ingredients','inventory_txn','labor_shift','AI Insights','AI Assistant','Shopping List'
    ])

    if menu == 'ingredients':
        st.title('🧾 Ingredients')
        df = get_table('ingredients')
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save Changes'):
            ed = ed.apply(calculate_cost_columns, axis=1)
            persist('ingredients')
            st.success('Saved')
    elif menu == 'recipes':
        st.title('📖 Recipes')
        df = get_table('recipes')
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save Recipes'):
            persist('recipes')
            st.success('Saved')
    elif menu == 'recipe_ingredients':
        st.warning('Use AI Assistant to manage recipes.')
    elif menu == 'inventory_txn':
        st.warning('Inventory transactions coming soon.')
    elif menu == 'labor_shift':
        st.warning('Labor shifts coming soon.')
    elif menu == 'AI Insights':
        st.title('🤖 AI Insights')
        df = get_table('ingredients')
        if st.button('Generate Insights'):
            st.markdown(ai_handle('analyze costs'))
    elif menu == 'AI Assistant':
        st.title('🤖 Assistant')
        for c in st.session_state['chat_log']:
            st.chat_message(c['role']).markdown(c['text'])
        audio = st.file_uploader('Upload voice (wav/mp3)', type=['wav','mp3'])
        prompt = transcribe_audio(audio) if audio else st.chat_input('Type command...')
        if prompt:
            st.session_state['chat_log'].append({'role':'user','text':prompt})
            res = ai_handle(prompt)
            st.session_state['chat_log'].append({'role':'assistant','text':res})
            save_table('recipes', get_table('recipes'))
            save_table('recipe_ingredients', get_table('recipe_ingredients'))
    elif menu == 'Shopping List':
        st.markdown(ai_handle('shopping list'))
    st.sidebar.markdown('---')
    st.sidebar.markdown('Made with ❤️')

# Self-tests
if __name__=='__main__' and not HAS_STREAMLIT:
    ing = load_table('ingredients'); assert isinstance(ing, pd.DataFrame)
    row = pd.Series({'purchase_unit':'g','purchase_qty':100,'purchase_price':2,'yield_percent':100})
    calc = calculate_cost_columns(row.copy()); assert calc['cost_per_gram']==0.02
    print('All tests passed')
