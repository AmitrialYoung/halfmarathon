import streamlit as st 
import json
import os
import joblib
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from langfuse import observe
from langfuse.openai import OpenAI 

load_dotenv()

@st.cache_resource
def load_model_locally():
    # Model jest ładowany bezpośrednio z githuba
    model_path = "model/new/best_marathon_model.pkl"
    model = joblib.load(model_path)
    return model

model = load_model_locally()

st.set_page_config(page_title="Predyktor biegu", layout="centered")
st.title("Predykcja czasu półmaratonu Wrocławskiego na podstawie czasu biegu na 5 km.")

# Pole do wprowadzenia klucza OpenAI w pasku bocznym
with st.sidebar:
    st.header("Konfiguracja")
    user_api_key = st.text_input("Podaj swój OpenAI API Key:", type="password")

text = st.text_area("Opisz siebie (imię/płeć, wiek, czas na 5 km):", height=150)


SYSTEM_PROMPT = "Jesteś silnikiem ekstrakcji danych. Zwracasz wyłącznie poprawny JSON."
USER_PROMPT_TEMPLATE = """
Z tekstu wyciągnij dane biegacza i ZNORMALIZUJ je do postaci:

- age: liczba całkowita
- gender: 'M' albo 'K'
- time_5km: ZAWSZE w formacie HH:MM:SS (np. 00:27:33)

Użytkownik może podać czas w dowolnej formie (np. "27:33", "27 minut", "0:27:33", "pół godziny").
Twoim zadaniem jest przeliczyć i zwrócić poprawne HH:MM:SS.

Tekst:
{text}

Zwróć wyłącznie JSON:
{{"age":..., "gender":..., "time_5km":...}}
Braki uzupełnij null.
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
def extract_runner_data(text, api_key):
    # Inicjalizacja klienta OpenAI z kluczem użytkownika przy każdym wywołaniu
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ]
    )

    content = response.choices[0].message.content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    clean_json = content[start:end]

    return json.loads(clean_json)

if st.button("Oblicz przewidywany czas półmaratonu"):
    # Dodatkowa walidacja obecności klucza API
    if not user_api_key:
        st.error("Proszę podać klucz OpenAI API w panelu bocznym.")
    elif not text.strip():
        st.error("Wpisz opis.")
    else:
        try:
            # Przekazanie klucza do funkcji ekstrakcji
            data = extract_runner_data(text, user_api_key)
            time_sec = time_to_seconds(data["time_5km"])

            missing = []
            age = data.get("age")
            gender = data.get("gender")

            if age is None: missing.append("wiek")
            if gender is None: missing.append("płeć")
            if time_sec is None: missing.append("czas 5 km")

            if missing:
                st.error("Brakuje danych: " + ", ".join(missing))
                st.stop()

            if not (18 <= age <= 99):
                st.error("Wiek musi być między 18 a 99 lat.")
                st.stop()

            # Budowa ramki danych dla modelu
            X = pd.DataFrame([{
                "Wiek": int(age),
                "Płeć": gender,
                "5 km Czas": float(time_sec)
            }])

            predicted_seconds = float(model.predict(X)[0])
            predicted_time = seconds_to_hms(predicted_seconds)

            st.success(f"Przewidywany czas półmaratonu: **{predicted_time}**")

        except Exception as e:
            st.error("Błąd przetwarzania danych lub predykcji. Upewnij się, że klucz API jest poprawny.")
            st.exception(e)