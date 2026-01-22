import streamlit as st 
import json
import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from langfuse import observe
from langfuse.openai import OpenAI

load_dotenv()

# Funkcja ładująca model z GitHub
@st.cache_resource
def load_model_locally():
    model_path = "model/new/best_marathon_model.pkl"
    # Wczytanie modelu za pomocą joblib
    model = joblib.load(model_path)
    return model

# Inicjalizacja modelu przy starcie aplikacji
model = load_model_locally()

st.set_page_config(page_title="Predyktor biegu", layout="centered")

# Sidebar do wprowadzania klucza API
with st.sidebar:
    st.header("Konfiguracja")
    # Użytkownik wpisuje swój klucz, który nie jest zapisywany w zmiennych systemowych
    user_api_key = st.text_input("Podaj swój OpenAI API Key:", type="password")
    
    if not user_api_key:
        st.warning("⚠️ Wprowadź klucz API w tym polu, aby aplikacja mogła przetworzyć Twój opis.")

st.title("Predykcja czasu półmaratonu Wrocławskiego")
st.markdown("Wpisz swój wiek, płeć oraz czas na 5 km, a AI i model ML obliczą Twój wynik.")

# Pole do wpisywania danych tekstowych
text = st.text_area("Opisz siebie (imię/płeć, wiek, czas na 5 km):", height=150, placeholder="Jestem mężczyzną, mam 34 lata. Mój ostatni czas na 5 km to 24:15.")
st.markdown("*Model nie jest doskonały. Błąd w predykcji oscyluje w granicach 5 minut.*")

# Instrukcje dla modelu LLM
SYSTEM_PROMPT = "Jesteś silnikiem ekstrakcji danych. Zwracasz wyłącznie poprawny JSON."
USER_PROMPT_TEMPLATE = """
Z tekstu wyciągnij dane biegacza i ZNORMALIZUJ je do postaci:
- age: liczba całkowita
- gender: 'M' albo 'K'
- time_5km: ZAWSZE w formacie HH:MM:SS (np. 00:27:33)

Tekst użytkownika:
{text}

Zwróć wyłącznie JSON:
{{"age":..., "gender":..., "time_5km":...}}
Braki uzupełnij null.
"""

# Funkcje pomocnicze do przeliczania czasu
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

# FUNKCJA EKSTRAKCJI: @observe() nie widzi klucza API, bo nie jest on argumentem funkcji
@observe()
def extract_runner_data(text_input, api_key_internal):
    # Klient OpenAI wewnątrz funkcji
    client = OpenAI(api_key=api_key_internal)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text_input)}
        ]
    )

    content = response.choices[0].message.content.strip()
    # Wyłuskanie czystego JSON
    start = content.find("{")
    end = content.rfind("}") + 1
    return json.loads(content[start:end])

# GŁÓWNA LOGIKA APLIKACJI
if st.button("Oblicz przewidywany czas półmaratonu"):
    if not user_api_key:
        st.error("Błąd: Brak klucza OpenAI API w panelu bocznym.")
    elif not text.strip():
        st.error("Błąd: Wpisz opis biegu.")
    else:
        try:
            with st.spinner("LLM analizuje tekst..."):
                # Wywołanie ekstrakcji - klucz API jest przekazywany bezpośrednio
                data = extract_runner_data(text, user_api_key)
            
            # Pobranie wyekstrahowanych danych
            age = data.get("age")
            gender = data.get("gender")
            time_5km_str = data.get("time_5km")
            time_sec = time_to_seconds(time_5km_str) if time_5km_str else None

            # Sprawdzenie czy LLM znalazł wszystkie dane
            missing = []
            if age is None: missing.append("wiek")
            if gender is None: missing.append("płeć")
            if time_sec is None: missing.append("czas na 5 km")

            if missing:
                st.error(f"Nie udało się odczytać: {', '.join(missing)}. Spróbuj opisać dane dokładniej.")
                st.stop()

            # Walidacja wieku dla modelu
            if not (18 <= int(age) <= 99):
                st.error("Model obsługuje wiek od 18 do 99 lat.")
                st.stop()

            # Danych do predykcji
            X = pd.DataFrame([{
                "Wiek": int(age),
                "Płeć": gender,
                "5 km Czas": float(time_sec)
            }])

            # Wykonanie predykcji modelem ML
            with st.spinner("Model ML oblicza wynik..."):
                predicted_seconds = float(model.predict(X)[0])
                predicted_time = seconds_to_hms(predicted_seconds)

            # Wyświetlenie wyniku
            st.success(f"### Twój przewidywany czas: {predicted_time}")
            
            # Wyświetlenie danych pomocniczych w sidebarze dla transparentności
            with st.sidebar:
                st.divider()
                st.write("**Dane przekazane do modelu:**")
                st.json(data)

        except Exception as e:
            st.error("Wystąpił błąd. Sprawdź czy Twój klucz API jest poprawny.")