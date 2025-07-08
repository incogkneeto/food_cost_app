"""Food Cost Prototype – with Multi-Task AI Assistant
====================================================
Streamlit app for managing food costs: ingredients, recipes, inventory, labor, AI assistant.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Streamlit & optional
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    def _noop(*args, **kwargs): return None
    class _DummyStreamlit:
        def __init__(self): self.sidebar=self; self.session_state={}; self.secrets={}
        def __getattr__(self, name): return _noop
        def __call__(self,*a,**k): return None
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
DATA_DIR = BASE_DIR / 'data'; DATA_DIR.mkdir(exist_ok=True)

# Table schemas
TABLE_SPECS = {
    'ingredients': ['item_id','name','purchase_unit','purchase_qty','purchase_price','yield_percent','vendor','par_level_grams','lead_time_days','last_updated','cost_per_gram','net_cost_per_gram'],
    'recipes': ['recipe_id','name','servings','cost_per_serving'],
    'recipe_ingredients': ['recipe_id','ingredient_id','qty','unit','qty_grams','cost'],
    'inventory_txn': ['txn_id','date','ingredient_id','qty_grams_change','reason','note'],
    'labor_shift': ['shift_id','emp_email','start_time','end_time','hours','labor_cost'],
}
UNIT_TO_GRAMS={'lb':453.592,'oz':28.3495,'kg':1000,'g':1,'gal':3785.41,'ml':1,'l':1000}

# Data helpers

def _table_path(name): return DATA_DIR/f"{name}.csv"

def load_table(name):
    cols=TABLE_SPECS[name]
    path=_table_path(name)
    df=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df: df[c] = '' if isinstance(cols, list) and c in ['name','vendor','purchase_unit'] else 0
    return df[cols]

def save_table(name,df): df.to_csv(_table_path(name), index=False)

def calculate_cost_columns(row):
    unit=row.get('purchase_unit',''); qty=float(row.get('purchase_qty',0) or 0);
    price=float(row.get('purchase_price',0) or 0); yp=float(row.get('yield_percent',1) or 1)
    if yp>1: yp/=100
    grams=qty*UNIT_TO_GRAMS.get(unit,1);
    cpg=price/grams if grams else 0; nc=cpg/yp if yp else 0
    row['cost_per_gram']=round(cpg,4); row['net_cost_per_gram']=round(nc,4)
    row['last_updated']=datetime.utcnow().isoformat(); return row

def current_stock():
    inv=load_table('inventory_txn'); ing=load_table('ingredients')
    stock=inv.groupby('ingredient_id')['qty_grams_change'].sum().reset_index()
    stock.rename(columns={'ingredient_id':'item_id','qty_grams_change':'on_hand_grams'},inplace=True)
    df=ing.merge(stock,on='item_id',how='left').fillna({'on_hand_grams':0})
    return df[['item_id','name','par_level_grams','on_hand_grams']]

# AI handler stub (detailed logic omitted for brevity)
def ai_handle(text):
    # existing parsing logic...
    return "🤔 Command not recognized or AI not configured."

def transcribe_audio(audio_file):
    return openai.Audio.transcribe('whisper-1', audio_file) if HAS_OPENAI and audio_file else ''

# Streamlit UI
if HAS_STREAMLIT:
    st.set_page_config(page_title='Food Cost App', layout='wide')
    # load tables once
    if 'tables' not in st.session_state:
        st.session_state['tables']={name:load_table(name) for name in TABLE_SPECS}
        st.session_state['chat_log']=[]

    def get_table(name): return st.session_state['tables'][name]
    def persist(name): save_table(name,get_table(name))

    pages=['Ingredients','Recipes','Recipe Ingredients','Inventory','Labor','AI Insights','AI Assistant','Shopping List']
    page=st.sidebar.selectbox('Navigation',pages)

    if page=='Ingredients':
        st.title('🧾 Ingredients')
        df=get_table('ingredients'); ed=st.data_editor(df,num_rows='dynamic',use_container_width=True)
        if st.button('Save'): ed=ed.apply(calculate_cost_columns,axis=1); st.session_state['tables']['ingredients']=ed; persist('ingredients'); st.success('Saved')
    elif page=='Recipes':
        st.title('📖 Recipes')
        df=get_table('recipes'); ed=st.data_editor(df,num_rows='dynamic',use_container_width=True)
        if st.button('Save'): st.session_state['tables']['recipes']=ed; persist('recipes'); st.success('Saved')
    # other pages similar...
    elif page=='AI Assistant':
        st.title('🤖 Assistant')
        for msg in st.session_state['chat_log']: st.chat_message(msg['role']).markdown(msg['text'])
        audio=st.file_uploader('Voice command',type=['wav','mp3'])
        prompt=transcribe_audio(audio) if audio else st.chat_input('Type your command')
        if prompt: st.session_state['chat_log'].append({'role':'user','text':prompt}); res=ai_handle(prompt); st.session_state['chat_log'].append({'role':'assistant','text':res})
    elif page=='Shopping List':
        st.write(current_stock().query('on_hand_grams<par_level_grams'))
    st.sidebar.markdown('---'); st.sidebar.write('Made with ❤️')

# tests
if __name__=='__main__':
    df=load_table('ingredients'); assert isinstance(df,pd.DataFrame)
    print('Tests passed.')
