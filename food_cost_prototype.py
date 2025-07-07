"""Food Cost Prototype – with Voice Command Support
====================================================
Adds file upload + basic microphone capture for voice commands via OpenAI Whisper.
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
    st = _DummyStreamlit()  # type: ignore
    HAS_STREAMLIT = False

try:
    import openai
    HAS_OPENAI = True
except ModuleNotFoundError:
    openai = None  # type: ignore
    HAS_OPENAI = False

# Whisper for transcription
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Configuration constants
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Table specifications for CSV storage
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
# Unit conversion mapping
UNIT_TO_GRAMS = {
    "lb": 453.592,
    "oz": 28.3495,
    "kg": 1000,
    "g": 1,
    "gal": 3785.41,
    "ml": 1,
    "l": 1000,
}

#----------------------------------------------------------------------------#
# Data helper functions                                                        #
#----------------------------------------------------------------------------#

def _table_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def load_table(name: str) -> pd.DataFrame:
    cols = TABLE_SPECS[name]["columns"]
    path = _table_path(name)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=cols)
    # Ensure all columns present
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
    cost_per_gram = price / grams if grams else 0.0
    net_cost = cost_per_gram / yp if yp else 0.0
    row["cost_per_gram"] = round(cost_per_gram, 4)
    row["net_cost_per_gram"] = round(net_cost, 4)
    row["last_updated"] = datetime.utcnow().isoformat()
    return row


def current_stock() -> pd.DataFrame:
    inv = (st.session_state.get("tables", {}).get("inventory_txn")
           if HAS_STREAMLIT and "inventory_txn" in st.session_state.get("tables", {})
           else load_table("inventory_txn"))
    ing = (st.session_state.get("tables", {}).get("ingredients")
           if HAS_STREAMLIT and "ingredients" in st.session_state.get("tables", {})
           else load_table("ingredients"))
    stock = inv.groupby("ingredient_id")["qty_grams_change"].sum().reset_index()
    stock.rename(columns={"ingredient_id":"item_id","qty_grams_change":"on_hand_grams"}, inplace=True)
    return ing[["item_id","name"]].merge(stock, on="item_id", how="left").fillna({"on_hand_grams": 0})

#----------------------------------------------------------------------------#
# AI command parser                                                            #
#----------------------------------------------------------------------------#

def ai_handle(text: str) -> str:
    t = text.lower().strip()
    # Stock check
    if t.startswith("how many") and "left" in t:
        nm = t.replace("how many", "").replace("left", "").strip()
        row = current_stock()[lambda df: df["name"].str.lower() == nm]
        if not row.empty:
            return f"{nm.title()} on hand: **{int(row.iloc[0]['on_hand_grams'])} g**"
    # Add purchase
    m = re.match(r"add\s+(\d+[.,]?\d*)\s*(lb|oz|kg|g)\s+([\w\s]+)\s*@\s*\$?(\d+[.,]?\d*)", t)
    if m and HAS_STREAMLIT:
        qty, unit, nm, pr = m.groups()
        qty, pr = float(qty.replace(",", ".")), float(pr.replace(",", "."))
        df = st.session_state["tables"]["ingredients"]
        existing = df[df["name"].str.lower() == nm.strip()]
        if existing.empty:
            new = {"item_id": f"NEW{len(df)+1}", "name": nm.title().strip(),
                   "purchase_unit": unit, "purchase_qty": qty, "purchase_price": pr, "yield_percent": 1.0}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        else:
            idx = existing.index[0]
            df.loc[idx, ["purchase_unit","purchase_qty","purchase_price"]] = [unit, qty, pr]
        df = df.apply(calculate_cost_columns, axis=1)
        st.session_state["tables"]["ingredients"] = df
        save_table("ingredients", df)
        return f"✅ Recorded {qty} {unit} {nm.title()} @ ${pr}"
    # GPT fallback
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a food-truck cost app assistant."},
                {"role":"user","content": text},
            ],
        )
        return resp.choices[0].message["content"].strip()
    return "🤔 Sorry, I couldn’t parse that."

#----------------------------------------------------------------------------#
# Audio transcription helper                                                   #
#----------------------------------------------------------------------------#

def transcribe_audio(audio_file) -> str:
    if not (HAS_OPENAI and audio_file):
        return ""
    return openai.Audio.transcribe("whisper-1", audio_file)

#----------------------------------------------------------------------------#
# AI Insights                                                                  #
#----------------------------------------------------------------------------#

def get_ai_insights(df: pd.DataFrame) -> str:
    if not (HAS_OPENAI and os.getenv("OPENAI_API_KEY")):
        return "AI not configured – set OPENAI_API_KEY to enable insights."
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a food-cost analyst."},
            {"role":"user","content": df.to_csv(index=False)},
        ],
    )
    return resp.choices[0].message["content"].strip()

#----------------------------------------------------------------------------#
# Streamlit UI
#----------------------------------------------------------------------------#
if HAS_STREAMLIT:
    st.set_page_config(page_title="Food Cost App", page_icon="🍔", layout="wide")
    # Initialize tables and chat log
    if "tables" not in st.session_state:
        st.session_state["tables"] = {name: load_table(name) for name in TABLE_SPECS}
        st.session_state["chat_log"] = []

    def get_table(name: str) -> pd.DataFrame:
        return st.session_state["tables"][name]
    def persist(name: str) -> None:
        save_table(name, get_table(name))

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Ingredients",
            "Recipes",
            "Sales (log)",
            "Inventory Counts",
            "Labor Shifts",
            "AI Insights",
            "AI Assistant",
        ],
    )

    if menu == "Ingredients":
        st.title("🧾 Ingredients")
        df = get_table("ingredients")
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Changes"):
            ed = ed.apply(calculate_cost_columns, axis=1)
            st.session_state["tables"]["ingredients"] = ed
            persist("ingredients")
            st.success("Saved")

    elif menu == "Recipes":
        st.title("📖 Recipes")
        df = get_table("recipes")
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Recipes"):
            st.session_state["tables"]["recipes"] = ed
            persist("recipes")
            st.success("Saved")

    elif menu == "Sales (log)":
        st.title("💵 Sales Log")
        st.warning("Stub: connect sales to inventory.")

    elif menu == "Inventory Counts":
        st.title("📦 Inventory Counts")
        st.warning("Stub: add count form and variance.")

    elif menu == "Labor Shifts":
        st.title("⏱️ Labor Shifts")
        df = get_table("labor_shift")
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Shifts"):
            st.session_state["tables"]["labor_shift"] = ed
            persist("labor_shift")
            st.success("Saved")

    elif menu == "AI Insights":
        st.title("🤖 AI Insights")
        df = get_table("ingredients")
        if df.empty:
            st.info("Add ingredients first.")
        elif st.button("Generate Insights"):
            st.markdown(get_ai_insights(df))

    elif menu == "AI Assistant":
        st.title("🤖 Voice & Text Assistant")
        for chat in st.session_state["chat_log"]:
            st.chat_message(chat["role"]).markdown(chat["text"])
        audio = st.file_uploader("Upload voice command (wav/mp3)", type=["wav","mp3"])
        if audio and HAS_OPENAI:
            st.audio(audio)
            prompt = transcribe_audio(audio)
            st.markdown(f"**Transcribed:** {prompt}")
        else:
            prompt = st.chat_input("Type or speak your command...")
        if prompt:
            st.session_state["chat_log"].append({"role":"user","text":prompt})
            response = ai_handle(prompt)
            st.session_state["chat_log"].append({"role":"assistant","text":response})
            st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️")

# Self-tests
if __name__ == "__main__" and not HAS_STREAMLIT:
    ing = load_table("ingredients")
    assert isinstance(ing, pd.DataFrame)
    test_row = pd.Series({"purchase_unit":"g","purchase_qty":100,"purchase_price":2,"yield_percent":100})
    calc = calculate_cost_columns(test_row.copy())
    assert calc["cost_per_gram"] == 0.02
    print("All tests passed")
