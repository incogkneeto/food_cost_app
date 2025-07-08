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

# Streamlit & fallback
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    def _noop(*args, **kwargs): return None
    class _DummyStreamlit:
        def __init__(self): self.sidebar=self; self.session_state={}; self.secrets={}
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
DATA_DIR = BASE_DIR / 'data'; DATA_DIR.mkdir(exist_ok=True)

# Table schemas
TABLE_SPECS = {
    'ingredients': ['item_id','name','purchase_unit','purchase_qty','purchase_price','yield_percent','vendor','par_level_grams','lead_time_days','last_updated','cost_per_gram','net_cost_per_gram'],
    'recipes': ['recipe_id','name','servings','cost_per_serving'],
    'recipe_ingredients': ['recipe_id','ingredient_id','qty','unit','qty_grams','cost'],
    'inventory_txn': ['txn_id','date','ingredient_id','qty_grams_change','reason','note'],
    'labor_shift': ['shift_id','emp_email','start_time','end_time','hours','labor_cost'],
}
UNIT_TO_GRAMS = {'lb':453.592,'oz':28.3495,'kg':1000,'g':1,'gal':3785.41,'ml':1,'l':1000}

# Data helpers

def _table_path(name:str)->Path:
    return DATA_DIR/f"{name}.csv"

def load_table(name:str)->pd.DataFrame:
    cols=TABLE_SPECS[name]
    path=_table_path(name)
    df=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = '' if c in ['name','vendor','purchase_unit'] else 0
    return df[cols]

def save_table(name:str,df:pd.DataFrame)->None:
    df.to_csv(_table_path(name),index=False)

def calculate_cost_columns(row:pd.Series)->pd.Series:
    unit=row.get('purchase_unit','')
    qty=float(row.get('purchase_qty',0) or 0)
    price=float(row.get('purchase_price',0) or 0)
    yp=float(row.get('yield_percent',1) or 1)
    if yp>1: yp/=100
    grams=qty*UNIT_TO_GRAMS.get(unit,1)
    cpg=price/grams if grams else 0
    ncpg=cpg/yp if yp else 0
    row['cost_per_gram']=round(cpg,4)
    row['net_cost_per_gram']=round(ncpg,4)
    row['last_updated']=datetime.utcnow().isoformat()
    return row

def current_stock()->pd.DataFrame:
    inv=load_table('inventory_txn')
    ing=load_table('ingredients')
    stock=inv.groupby('ingredient_id')['qty_grams_change'].sum().reset_index()
    stock.rename(columns={'ingredient_id':'item_id','qty_grams_change':'on_hand_grams'},inplace=True)
    df=ing.merge(stock,on='item_id',how='left').fillna({'on_hand_grams':0})
    return df[['item_id','name','par_level_grams','on_hand_grams']]

# AI parse and multitask

def ai_handle(text:str)->str:
    t=text.lower().strip()
    # 1) Add ingredients from any recipe
    m = re.match(r'add ingredients from\s+"?(.+?)"?(?:\s+to ingredients)?', text, re.I)
    if m:
        name=m.group(1).strip()
        recipes=st.session_state.get('tables',{}).get('recipes',load_table('recipes'))
        row=recipes[recipes['name'].str.lower()==name.lower()]
        if row.empty:
            return f"❌ Recipe '{name}' not found."
        rid=row.iloc[0]['recipe_id']
        ri=st.session_state.get('tables',{}).get('recipe_ingredients',load_table('recipe_ingredients'))
        items=ri[ri['recipe_id']==rid]
        ing_df=st.session_state.get('tables',{}).get('ingredients',load_table('ingredients'))
        added=0
        for _,rec in items.iterrows():
            item_id=rec['ingredient_id']
            grams=rec['qty_grams']
            names=ing_df.loc[ing_df['item_id']==item_id,'name'].tolist()
            ing_name=names[0] if names else ''
            ex=ing_df[ing_df['name'].str.lower()==ing_name.lower()]
            if ex.empty:
                new={
                    'item_id':f'NEW{len(ing_df)+1}',
                    'name':ing_name.title(),
                    'purchase_unit':'',
                    'purchase_qty':grams,
                    'purchase_price':0,
                    'yield_percent':100,
                    'vendor':'',
                    'par_level_grams':0,
                    'lead_time_days':0
                }
                ing_df=pd.concat([ing_df,pd.DataFrame([new])],ignore_index=True)
            else:
                idx=ex.index[0]
                if not ing_df.at[idx,'purchase_unit']:
                    ing_df.at[idx,'purchase_unit']='g'
                if not ing_df.at[idx,'purchase_qty']:
                    ing_df.at[idx,'purchase_qty']=grams
                if not ing_df.at[idx,'yield_percent']:
                    ing_df.at[idx,'yield_percent']=100
            added+=1
        st.session_state['tables']['ingredients']=ing_df
        save_table('ingredients',ing_df)
        return f"✅ Added {added} ingredients from recipe '{name}' to Ingredients list."
    # 2) Shopping list
    if 'shopping list' in t:
        stock=current_stock()
        low=stock[stock['on_hand_grams']<stock['par_level_grams']]
        return '🛒 Shopping list: ' + ', '.join(low['name'].tolist()) if not low.empty else '🛒 No items below par level.'
    # 3) Stock check
    if t.startswith('how many') and 'left' in t:
        nm=t.replace('how many','').replace('left','').strip()
        row=current_stock()[lambda df:df['name'].str.lower()==nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # 4) Add purchase
    m2=re.match(r'add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)',text.lower())
    if m2 and HAS_STREAMLIT:
        qty,unit,nm,pr=m2.groups(); qty,pr=float(qty.replace(',','.')),float(pr.replace(',','.'))
        df=st.session_state['tables']['ingredients']
        ex=df[df['name'].str.lower()==nm.strip().lower()]
        if ex.empty:
            new={'item_id':f'NEW{len(df)+1}','name':nm.title(),'purchase_unit':unit,'purchase_qty':qty,'purchase_price':pr,'yield_percent':100}
            df=pd.concat([df,pd.DataFrame([new])],ignore_index=True)
        else:
            idx=ex.index[0]
            df.loc[idx,['purchase_unit','purchase_qty','purchase_price','yield_percent']]=[unit,qty,pr,100]
        df=df.apply(calculate_cost_columns,axis=1)
        st.session_state['tables']['ingredients']=df; save_table('ingredients',df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    # 5) GPT fallback
    if HAS_OPENAI and openai.api_key:
        resp=openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[{'role':'system','content':'You are a food-truck cost app assistant.'},{'role':'user','content':text}]
        )
        return resp.choices[0].message['content'].strip()
    return "🤔 Sorry, I couldn’t parse that."

# Transcribe helper

def transcribe_audio(audio_file):
    return openai.Audio.transcribe('whisper-1',audio_file) if HAS_OPENAI and audio_file else ''

# Streamlit UI
if HAS_STREAMLIT:
    st.set_page_config(page_title='Food Cost App',layout='wide')
    if 'tables' not in st.session_state:
        st.session_state['tables']={n:load_table(n) for n in TABLE_SPECS}
        st.session_state['chat_log']=[]
    def get_table(n): return st.session_state['tables'][n]
    def persist(n): save_table(n,get_table(n))
    pages=['Ingredients','Recipes','Recipe Ingredients','Inventory','Labor','AI Insights','AI Assistant','Shopping List']
    page=st.sidebar.selectbox('Navigation',pages)
    if page=='Ingredients':
        st.title('🧾 Ingredients'); df=get_table('ingredients'); ed=st.data_editor(df,num_rows='dynamic',use_container_width=True)
        if st.button('Save'): ed=ed.apply(calculate_cost_columns,axis=1); st.session_state['tables']['ingredients']=ed; persist('ingredients'); st.success('Saved')
    elif page=='AI Assistant':
        st.title('🤖 Assistant')
        for msg in st.session_state['chat_log']: st.chat_message(msg['role']).markdown(msg['text'])
        audio=st.file_uploader('Voice command',type=['wav','mp3'])
        prompt=transcribe_audio(audio) if audio else st.chat_input('Type your command')
        if prompt: st.session_state['chat_log'].append({'role':'user','text':prompt}); res=ai_handle(prompt); st.session_state['chat_log'].append({'role':'assistant','text':res})
    elif page=='Shopping List':
        st.write(current_stock().query('on_hand_grams<par_level_grams'))
    st.sidebar.markdown('---'); st.sidebar.write('Made with ❤️')
# Self-tests
if __name__=='__main__': df=load_table('ingredients'); assert isinstance(df,pd.DataFrame); print('Tests passed.')
