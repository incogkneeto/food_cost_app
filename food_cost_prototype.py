"""Food Cost Prototype – with Recipe Creation and Voice Command Support
====================================================
Adds voice/text AI assistant capable of parsing free-form recipe and ingredient commands.
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
    stock.rename(columns={"ingredient_id": "item_id", "qty_grams_change": "on_hand_grams"}, inplace=True)
    return ing[["item_id","name"]].merge(stock, on="item_id", how="left").fillna({"on_hand_grams": 0})

# AI parser & recipe creation

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    # Recipe creation
    if "recipe" in t:
        m = re.search(r'recipe\s+"([^"]+)"', text)
        name = m.group(1) if m else re.search(r'recipe\s+(\w[\w\s]+?)\s+with', text, re.I).group(1)
        sm = re.search(r'servings\s*(\d+)', text, re.I)
        servings = int(sm.group(1)) if sm else 1
        rdf = st.session_state["tables"]["recipes"]
        if not rdf.empty:
            ids = [int(r.replace('R','')) for r in rdf['recipe_id']]
            rid = f'R{max(ids)+1}'
        else:
            rid = 'R1'
        total_cost = 0
        ing_text = text.split('with',1)[1].split(';')[0]
        items = [i.strip() for i in re.split(r',|\n|- ', ing_text) if i.strip()]
        rif = []
        for item in items:
            m2 = re.match(r'(\d+[.,]?\d*)\s*(\w+)\s+(.+)', item)
            if not m2:
                continue
            q, u, nm = m2.groups()
            q = float(q.replace(',', '.'))
            grams = q * UNIT_TO_GRAMS.get(u, 1)
            ingdf = st.session_state["tables"]["ingredients"]
            er = ingdf[ingdf['name'].str.lower() == nm.strip().lower()]
            if not er.empty:
                idf = er.iloc[0]['item_id']
                cpf = float(er.iloc[0]['cost_per_gram'])
            else:
                idf = ''
                cpf = 0
            cost = cpf * grams
            total_cost += cost
            rif.append({'recipe_id': rid, 'ingredient_id': idf, 'qty': q, 'unit': u, 'qty_grams': grams, 'cost': cost})
        ridf = pd.DataFrame(rif)
        rifull = st.session_state["tables"]["recipe_ingredients"]
        st.session_state["tables"]["recipe_ingredients"] = pd.concat([rifull, ridf], ignore_index=True)
        cps = total_cost / servings
        newr = {'recipe_id': rid, 'name': name, 'servings': servings, 'cost_per_serving': cps}
        st.session_state["tables"]["recipes"] = pd.concat([rdf, pd.DataFrame([newr])], ignore_index=True)
        save_table("recipes", st.session_state["tables"]["recipes"])
        save_table("recipe_ingredients", st.session_state["tables"]["recipe_ingredients"])
        return f'✅ Recipe "{name}" added ({len(rif)} ingredients), cost per serving ${cps:.2f}'
    # Stock check
    if t.startswith("how many") and "left" in t:
        nm = t.replace("how many", "").replace("left", "").strip()
        row = current_stock()[lambda df: df['name'].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # Add purchase
    m3 = re.match(r"add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)", t)
    if m3 and HAS_STREAMLIT:
        qty, unit, nm, pr = m3.groups()
        qty = float(qty.replace(',', '.'))
        pr = float(pr.replace(',', '.'))
        df = st.session_state['tables']['ingredients']
        ex = df[df['name'].str.lower() == nm.strip().lower()]
        if ex.empty:
            new = {'item_id': f'NEW{len(df)+1}', 'name': nm.title(), 'purchase_unit': unit, 'purchase_qty': qty, 'purchase_price': pr, 'yield_percent': 1.0}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        else:
            idx = ex.index[0]
            df.loc[idx, ['purchase_unit','purchase_qty','purchase_price']] = [unit, qty, pr]
        df = df.apply(calculate_cost_columns, axis=1)
        st.session_state['tables']['ingredients'] = df
        save_table('ingredients', df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    # GPT fallback
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

# AI insights

def get_ai_insights(df: pd.DataFrame) -> str:
    if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
        return ai_handle('analyze costs')
    return 'AI not configured.'

# Streamlit UI
if HAS_STREAMLIT:
    st.set_page_config(page_title='Food Cost App', page_icon='🍔', layout='wide')
    if 'tables' not in st.session_state:
        st.session_state['tables'] = {n: load_table(n) for n in TABLE_SPECS}
        st.session_state['chat_log'] = []
    def get_table(name: str) -> pd.DataFrame:
        return st.session_state['tables'][name]
    def persist(name: str) -> None:
        save_table(name, get_table(name))
    menu = st.sidebar.radio('Navigation', list(TABLE_SPECS.keys()) + ['AI Insights','AI Assistant'])
    if menu == 'ingredients':
        st.title('🧾 Ingredients')
        df = get_table('ingredients')
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save Changes'):
            ed = ed.apply(calculate_cost_columns, axis=1)
            st.session_state['tables']['ingredients'] = ed
            persist('ingredients')
            st.success('Saved')
    elif menu == 'recipes':
        st.title('📖 Recipes')
        df = get_table('recipes')
        ed = st.data_editor(df, num_rows='dynamic', use_container_width=True)
        if st.button('Save Recipes'):
            st.session_state['tables']['recipes'] = ed
            persist('recipes')
            st.success('Saved')
    elif menu == 'recipe_ingredients':
        st.warning('Use AI Assistant to manage recipes.')
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
            save_table('recipes', st.session_state['tables']['recipes'])
            save_table('recipe_ingredients', st.session_state['tables']['recipe_ingredients'])
    st.sidebar.markdown('---')
    st.sidebar.markdown('Made with ❤️')

# Self-tests
if __name__ == '__main__' and not HAS_STREAMLIT:
    ing = load_table('ingredients')
    assert isinstance(ing, pd.DataFrame)
    test_row = pd.Series({'purchase_unit':'g','purchase_qty':100,'purchase_price':2,'yield_percent':100})
    calc = calculate_cost_columns(test_row.copy())
    assert calc['cost_per_gram'] == 0.02
    print('All tests passed')
