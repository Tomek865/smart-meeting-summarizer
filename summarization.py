import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def generate_summary(combined_script):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Brak klucza GROQ_API_KEY w pliku .env!")
        return None

    client = Groq(api_key=api_key)
    
    transcript_text = ""
    for line in combined_script:
        transcript_text += f"[{line['start_str']} - {line['end_str']}] {line['speaker']}: {line['text']}\n"

    system_prompt = """
    Jesteś profesjonalnym asystentem biznesowym. Otrzymasz transkrypcję ze spotkania z podziałem na mówców.
    Twoim zadaniem jest wygenerowanie zwięzłej, użytecznej notatki w języku polskim.
    
    Format wyjściowy musi zawierać 3 sekcje:
    1. **Cel spotkania:** (Krótkie streszczenie, o czym rozmawiano - max 2 zdania).
    2. **Kluczowe ustalenia:** (Najważniejsze punkty rozmowy w formie punktatorów).
    3. **Zadania (Action Items):** (Kto ma co zrobić, jeśli wynika to z kontekstu. Jeśli nie, napisz "Brak przypisanych zadań").
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Oto transkrypcja:\n\n{transcript_text}"}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.5,
        )
        
        result = response.choices[0].message.content
        
        if not result or result.strip() == "":
            st.error("API Groq zwróciło pustą odpowiedź. Spróbuj ponownie.")
            return None
            
        return result

    except Exception as e:
        st.error(f"Szczegóły błędu API Groq: {e}")
        print(f"\n--- BŁĄD GROQ API ---\n{e}\n---------------------\n")
        return None