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
    # 1) Add ingredients from any recipe (flexible commands)
    # e.g., "add ingredients from Carne Asada" or "add recipe Carne Asada to ingredients"
    m1 = re.match(r'add ingredients from\s+"?(.+?)"?$', text, re.I)
    m2 = re.match(r'add recipe\s+"?(.+?)"?\s+to ingredients$', text, re.I)
    m = m1 or m2
    if m:
        name = m.group(1).strip()
        recipes = st.session_state['tables'].get('recipes', load_table('recipes'))
        row = recipes[recipes['name'].str.lower() == name.lower()]
        if row.empty:
            return f"❌ Recipe '{name}' not found."
        rid = row.iloc[0]['recipe_id']
        ri = st.session_state['tables'].get('recipe_ingredients', load_table('recipe_ingredients'))
        items = ri[ri['recipe_id'] == rid]
        ing_df = st.session_state['tables'].get('ingredients', load_table('ingredients'))
        added = 0
        for _, rec in items.iterrows():
            item_id = rec['ingredient_id']
            grams = rec['qty_grams']
            names = ing_df.loc[ing_df['item_id'] == item_id, 'name'].tolist()
            ing_name = names[0] if names else ''
            ex = ing_df[ing_df['name'].str.lower() == ing_name.lower()]
            if ex.empty:
                new = {
                    'item_id': f'NEW{len(ing_df)+1}',
                    'name': ing_name.title(),
                    'purchase_unit': '',
                    'purchase_qty': grams,
                    'purchase_price': 0,
                    'yield_percent': 100,
                    'vendor': '',
                    'par_level_grams': 0,
                    'lead_time_days': 0
                }
                ing_df = pd.concat([ing_df, pd.DataFrame([new])], ignore_index=True)
            else:
                idx = ex.index[0]
                if not ing_df.at[idx, 'purchase_unit']:
                    ing_df.at[idx, 'purchase_unit'] = 'g'
                if not ing_df.at[idx, 'purchase_qty']:
                    ing_df.at[idx, 'purchase_qty'] = grams
                if not ing_df.at[idx, 'yield_percent']:
                    ing_df.at[idx, 'yield_percent'] = 100
            added += 1
        st.session_state['tables']['ingredients'] = ing_df
        save_table('ingredients', ing_df)
        return f"✅ Added {added} ingredients from recipe '{name}' to Ingredients list."
    # 2) Shopping list
    if 'shopping list' in t:
        stock = current_stock()
        low = stock[stock['on_hand_grams'] < stock['par_level_grams']]
        return '🛒 Shopping list: ' + ', '.join(low['name'].tolist()) if not low.empty else '🛒 No items below par level.'
    # 3) Stock check
    if t.startswith('how many') and 'left' in t:
        nm = t.replace('how many', '').replace('left', '').strip()
        row = current_stock()[lambda df: df['name'].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # 4) Add purchase
    m_add = re.match(r'add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)', text.lower())
    if m_add and HAS_STREAMLIT:
        qty, unit, nm2, pr = m_add.groups()
        qty, pr = float(qty.replace(',', '.')), float(pr.replace(',', '.'))
        df_ing = st.session_state['tables']['ingredients']
        ex2 = df_ing[df_ing['name'].str.lower() == nm2.strip().lower()]
        if ex2.empty:
            new = {
                'item_id': f'NEW{len(df_ing)+1}',
                'name': nm2.title(),
                'purchase_unit': unit,
                'purchase_qty': qty,
                'purchase_price': pr,
                'yield_percent': 100
            }
            df_ing = pd.concat([df_ing, pd.DataFrame([new])], ignore_index=True)
        else:
            idx2 = ex2.index[0]
            df_ing.loc[idx2, ['purchase_unit','purchase_qty','purchase_price','yield_percent']] = [unit, qty, pr, 100]
        df_ing = df_ing.apply(calculate_cost_columns, axis=1)
        st.session_state['tables']['ingredients'] = df_ing
        save_table('ingredients', df_ing)
        return f"✅ Recorded {qty} {unit} {nm2.title()} @ ${pr}"
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
if HAS_STREAMLIT:
    st.set_page_config(page_title='Food Cost App', layout='wide')
    if 'tables' not in st.session_state:
        st.session_state['tables'] = {n: load_table(n) for n in TABLE_SPECS}
        st.session_state['chat_log'] = []

    def get_table(n): return st.session_state['tables'][n]
    def persist(n): save_table(n, get_table(n))

    pages = ['Ingredients','Recipes','Recipe Ingredients','Inventory','Labor','AI Insights','AI Assistant','Shopping List']
    page = st.sidebar.selectbox('Navigation', pages)

    if page == 'Ingredients':
        st.title('🧾 Ingredients')
        df = get_table('ingredients')
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save'):
            ed = ed.apply(calculate_cost_columns, axis=1)
            st.session_state['tables']['ingredients'] = ed
            persist('ingredients')
            st.success('Saved')
    elif page == 'Recipes':
        st.title('📖 Recipes')
        df = get_table('recipes')
        if df.empty:
            st.info("No recipes yet – use AI Assistant or add manually.")
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save'):
            st.session_state['tables']['recipes'] = ed
            persist('recipes')
            st.success('Recipes saved')
    elif page == 'Recipe Ingredients':
        st.title('📋 Recipe Ingredients')
        df = get_table('recipe_ingredients')
        if df.empty:
            st.info("No recipe ingredients yet – use AI Assistant to create recipes.")
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save'):
            st.session_state['tables']['recipe_ingredients'] = ed
            persist('recipe_ingredients')
            st.success('Recipe ingredients saved')
    elif page == 'Inventory':
        st.title('📦 Inventory Transactions')
        df = get_table('inventory_txn')
        if df.empty:
            st.info("No inventory transactions yet – record sales or counts to update inventory.")
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save'):
            st.session_state['tables']['inventory_txn'] = ed
            persist('inventory_txn')
            st.success('Inventory transactions saved')
    elif page == 'Labor':
        st.title('⏱️ Labor Shifts')
        df = get_table('labor_shift')
        if df.empty:
            st.info("No labor shifts yet – record your team’s shifts here.")
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save'):
            st.session_state['tables']['labor_shift'] = ed
            persist('labor_shift')
            st.success('Labor shifts saved')
    elif page == 'AI Insights':
        st.title('🤖 AI Insights')
        df = get_table('ingredients')
        if df.empty:
            st.info("Add ingredients first – then generate insights.")
        elif st.button('Generate Insights'):
            insights = ai_handle('analyze costs')
            st.markdown(insights)
    elif page == 'AI Assistant':
        st.title('🤖 Assistant')
        for msg in st.session_state['chat_log']:
            st.chat_message(msg['role']).markdown(msg['text'])
        audio = st.file_uploader('Voice command', type=['wav','mp3'])
        prompt = transcribe_audio(audio) if audio else st.chat_input('Type your command')
        if prompt:
            st.session_state['chat_log'].append({'role':'user','text':prompt})
            res = ai_handle(prompt)
            st.session_state['chat_log'].append({'role':'assistant','text':res})
    elif page == 'Shopping List':
        st.title('🛒 Shopping List')
        stock = current_stock()
        low = stock[stock['on_hand_grams'] < stock['par_level_grams']]
        st.dataframe(low[['name','on_hand_grams','par_level_grams']])

    st.sidebar.markdown('---')
    st.sidebar.write('Made with ❤️')

# Self-tests
if __name__=='__main__':
    df = load_table('ingredients')
    assert isinstance(df, pd.DataFrame)
    print('Tests passed.')
