import streamlit as st 
import json
import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from langfuse import observe
from langfuse.openai import OpenAI

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Funkcja ładująca model z lokalnego repozytorium Git
@st.cache_resource
def load_model_locally():
    model_path = "model/new/best_marathon_model.pkl"
    return joblib.load(model_path)

model = load_model_locally()

st.set_page_config(page_title="Predyktor biegu", layout="centered")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Konfiguracja")
    input_key = st.text_input("Podaj swój OpenAI API Key:", type="password")
    
    if input_key:
        st.session_state["user_provided_api_key"] = input_key
    else:
        st.warning("⚠️ Wprowadź klucz API, aby kontynuować.")

st.title("Predykcja czasu półmaratonu")

text = st.text_area("Opisz siebie (płeć, wiek, czas na 5 km):", height=150)
st.markdown("*Model nie jest doskonały, błąd predykcji oscyluje w granicach 5 minut*")

SYSTEM_PROMPT = "Jesteś silnikiem ekstrakcji danych. Zwracasz wyłącznie poprawny JSON."
USER_PROMPT_TEMPLATE = """
Z tekstu wyciągnij dane biegacza i ZNORMALIZUJ je do postaci:
- age: liczba całkowita
- gender: 'M' albo 'K'
- time_5km: ZAWSZE w formacie HH:MM:SS (np. 00:27:33)

Tekst: {text}

Zwróć wyłącznie JSON: {{"age":..., "gender":..., "time_5km":...}}
"""

def time_to_seconds(time_str):
    try:
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s
    except:
        return None

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@observe()
def extract_runner_data(text_input):
    # Pobieramy klucz API bezpośrednio z sesji Streamlit wewnątrz funkcji.
    # Dzięki temu @observe() widzi tylko 'text_input' w zakładce Input w Langfuse.
    api_key = st.session_state.get("user_provided_api_key")
    
    if not api_key:
        raise ValueError("Brak klucza API w sesji aplikacji.")

    # Inicjalizacja klienta z kluczem, który nie jest logowany jako argument funkcji
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text_input)}
        ]
    )

    content = response.choices[0].message.content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    return json.loads(content[start:end])

# --- LOGIKA PRZYCISKU ---
if st.button("Oblicz przewidywany czas"):
    if "user_provided_api_key" not in st.session_state:
        st.error("Podaj klucz API w panelu bocznym!")
    elif not text.strip():
        st.error("Wpisz opis biegu.")
    else:
        try:
            # Przekazujemy tylko tekst - Langfuse nie przechwyci klucza
            data = extract_runner_data(text)
            
            time_sec = time_to_seconds(data.get("time_5km"))
            age = data.get("age")
            gender = data.get("gender")

            if None in [time_sec, age, gender]:
                st.error("LLM nie wyekstrahował wszystkich danych. Spróbuj opisać się jaśniej.")
                st.stop()

            # Przygotowanie danych do modelu ML
            X = pd.DataFrame([{
                "Wiek": int(age),
                "Płeć": gender,
                "5 km Czas": float(time_sec)
            }])

            predicted_seconds = float(model.predict(X)[0])
            st.success(f"### Przewidywany wynik: {seconds_to_hms(predicted_seconds)}")

        except Exception as e:
            st.error("Wystąpił błąd. Sprawdź swój klucz API lub połączenie.")