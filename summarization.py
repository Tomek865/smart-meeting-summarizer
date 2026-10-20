import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def generate_summary(combined_script, summary_style="Biznesowy", custom_instructions="", language="Polski"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Brak klucza GROQ_API_KEY w pliku .env!")
        return None

    client = Groq(api_key=api_key)
    
    transcript_text = ""
    for line in combined_script:
        transcript_text += f"[{line['start_str']} - {line['end_str']}] {line['speaker']}: {line['text']}\n"

    if language == "Polski":
        lang_instruction = "w języku polskim"
    else:
        lang_instruction = "w języku angielskim (ALL OUTPUT MUST BE STRICTLY IN ENGLISH)"

    if summary_style == "Biznesowy":
        system_prompt = f"""
        Jesteś profesjonalnym asystentem biznesowym. Otrzymasz transkrypcję ze spotkania z podziałem na mówców.
        Twoim zadaniem jest wygenerowanie zwięzłej, użytecznej notatki {lang_instruction}.
        
        Format wyjściowy musi zawierać 3 sekcje:
        1. **Cel spotkania:** (Krótkie streszczenie, o czym rozmawiano - max 2 zdania).
        2. **Kluczowe ustalenia:** (Najważniejsze punkty rozmowy w formie punktatorów).
        3. **Zadania (Action Items):** (Kto ma co zrobić, jeśli wynika to z kontekstu. Jeśli nie, napisz "Brak przypisanych zadań").
        """
    elif summary_style == "Naukowy / Akademicki":
        system_prompt = f"""
        Jesteś asystentem naukowym. Otrzymasz transkrypcję z dyskusji, wywiadu badawczego lub wykładu.
        Twoim zadaniem jest przygotowanie rzetelnego, analitycznego podsumowania {lang_instruction}.
        
        Format wyjściowy musi zawierać 3 sekcje:
        1. **Główny problem / Teza:** (Jaki był główny temat badawczy lub problematyka rozmowy).
        2. **Argumenty i koncepcje:** (Wypunktowane kluczowe twierdzenia, definicje lub tezy uczestników).
        3. **Wnioski końcowe:** (Podsumowanie dyskusji i ewentualne kierunki dalszych rozważań).
        """
    elif summary_style == "Luźna notatka (Casual)":
        system_prompt = f"""
        Jesteś przyjaznym, pomocnym asystentem. Przeczytaj poniższą transkrypcję rozmowy i zrób z niej krótką, zwięzłą notatkę {lang_instruction}.
        Napisz w kilku zdaniach o czym gadano, wyciągnij najważniejsze wnioski w punktach i rzuć okiem, czy ktoś miał coś do zrobienia. Używaj potocznego, lekkiego języka bez sztywnego korpo-żargonu.
        """
    else:
        system_prompt = f"Jesteś pomocnym asystentem. Podsumuj poniższą transkrypcję {lang_instruction}."

    if custom_instructions and custom_instructions.strip() != "":
        system_prompt += f"\n\n--- UWAGA! DODATKOWE INSTRUKCJE OD UŻYTKOWNIKA (bezwzględnie się do nich zastosuj): ---\n{custom_instructions.strip()}"

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