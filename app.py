import streamlit as st 
import boto3
import json
import os
import joblib
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from langfuse import observe, Langfuse
from langfuse.openai import OpenAI

load_dotenv()
client = OpenAI()

BUCKET_NAME = "half-marathon"

@st.cache_resource
def load_model_from_spaces():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
    )

    obj = s3.get_object(
        Bucket=BUCKET_NAME,
        Key="model/new/best_marathon_model.pkl"
    )

    model = joblib.load(BytesIO(obj["Body"].read()))
    return model

model = load_model_from_spaces()

st.set_page_config(page_title="Predyktor biegu", layout="centered")
st.title("Predykcja czasu półmaratonu Wrocławskiego na podstawie czasu biegu na 5 km.")

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
def extract_runner_data(text):
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
    if not text.strip():
        st.error("Wpisz opis.")
    else:
        try:
            data = extract_runner_data(text)
            time_sec = time_to_seconds(data["time_5km"])

            missing = []

            age = data.get("age")
            gender = data.get("gender")

            if age is None:
                missing.append("wiek")
            if gender is None:
                missing.append("płeć")
            if time_sec is None:
                missing.append("czas 5 km")

            if missing:
                st.error("Brakuje danych: " + ", ".join(missing))
                st.stop()

            if not (18 <= age <= 99):
                st.error("Wiek musi być między 18 a 99 lat.")
                st.stop()

            model_json = {
                "Wiek": int(data["age"]),
                "Płeć": data["gender"],
                "Czas_5km_sekundy": int(time_sec)
            }

            with st.sidebar:
                st.title("Dane wejściowe do modelu")
                st.markdown(
                    "To są dane wyekstrahowane przez LLM i przekazane do modelu. "
                    "Czas 5 km jest zapisany w sekundach, ponieważ w takiej postaci trenowany był model."
                )
                st.code(json.dumps(model_json, ensure_ascii=False, indent=2), language="json")

            X = pd.DataFrame([{
                "Wiek": model_json["Wiek"],
                "Płeć": model_json["Płeć"],
                "5 km Czas": float(model_json["Czas_5km_sekundy"])
            }])

            predicted_seconds = float(model.predict(X)[0])
            predicted_time = seconds_to_hms(predicted_seconds)

            st.success(f"Przewidywany czas półmaratonu: **{predicted_time}**")

        except Exception as e:
            st.error("Błąd przetwarzania danych lub predykcji.")
            st.exception(e)
