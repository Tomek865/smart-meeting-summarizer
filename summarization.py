import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def generate_summary(combined_script, summary_style="Biznesowy", custom_instructions="", language="Polski"):
    """
    Generuje wyczerpujące podsumowanie transkrypcji przy użyciu LLM (Groq).
    Dostosowuje stopień szczegółowości do długości nagrania oraz wybranego języka (Polski / Angielski).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Brak klucza GROQ_API_KEY w pliku .env!")
        return None

    client = Groq(api_key=api_key)
    
    transcript_text = ""
    total_words = 0
    for line in combined_script:
        text = line.get("text", "").strip()
        total_words += len(text.split())
        transcript_text += f"[{line['start_str']} - {line['end_str']}] {line['speaker']}: {text}\n"

    is_english = "ang" in str(language).strip().lower() or "en" in str(language).strip().lower()
    is_long_audio = total_words > 350 or len(combined_script) > 15

    if is_english:
        lang_directive = "ALL OUTPUT MUST BE STRICTLY AND EXCLUSIVELY IN ENGLISH. ALL HEADINGS, LABELS, AND DESCRIPTIONS MUST BE IN ENGLISH."
        
        detail_guidance = (
            "EXTENSIVE & HIGHLY DETAILED SUMMARY REQUIRED:\n"
            "This recording contains a substantial, multi-topic discussion. Produce an exhaustive, granular report.\n"
            "- Do not settle for brief generalizations or superficial 1-2 sentence notes.\n"
            "- Cover every topic, argument, proposed solution, and decision in thorough detail.\n"
            "- Explicitly retain all specific numbers, financial values, metrics, project milestones, dates, and named stakeholders.\n"
            "- Develop deep bullet points with full context explaining *why* decisions were made."
            if is_long_audio else
            "DEPTH & QUALITY REQUIREMENT:\n"
            "Provide a thorough, comprehensive summary. Cover all discussed subjects without omitting key arguments, context, or decisions."
        )

        formatting_rules = (
            "FORMATTING GUIDELINES:\n"
            "- In bullet points, NEVER wrap entire sentences in double asterisks (e.g. NEVER write: '- **Entire sentence.**').\n"
            "- Use bold (**) strictly for a short label or person's name at the very beginning of a bullet (e.g. '- John: Task description' or '- Budget: Final decision').\n"
            "- Ensure clean, professional markdown."
        )

        if summary_style == "Biznesowy":
            system_prompt = f"""
You are an elite executive business assistant and meeting minutes specialist.
{lang_directive}

{detail_guidance}

{formatting_rules}

Generate a structured business report using the following sections:
1. Executive Summary & Context: (A thorough description of the meeting's purpose, background context, main problems analyzed, and strategic significance).
2. Comprehensive Discussion Points & Decisions: (A detailed, in-depth breakdown of each agenda item discussed, options evaluated, arguments from participants, and agreements reached).
3. Action Items & Next Steps: (A concrete, detailed list of tasks. Include task owners, expected outcomes, deadlines, and dependencies. If no tasks were assigned, explicitly note that).
"""
        elif summary_style == "Naukowy / Akademicki":
            system_prompt = f"""
You are an advanced academic research assistant and scientific analyst.
{lang_directive}

{detail_guidance}

{formatting_rules}

Generate a comprehensive academic report using the following sections:
1. Research Problem & Theoretical Context: (Detailed exposition of the research question, core problem, hypotheses, and conceptual background).
2. In-Depth Arguments, Concepts & Analysis: (Rigorous, detailed breakdown of each participant's claims, theoretical concepts, empirical evidence discussed, counterarguments, and analytical nuances).
3. Conclusions & Future Directions: (Comprehensive synthesis of findings, open scientific questions, and recommendations for future inquiry).
"""
        elif summary_style == "Luźna notatka (Casual)":
            system_prompt = f"""
You are a friendly, highly competent personal assistant.
{lang_directive}

{detail_guidance}

{formatting_rules}

Generate a detailed, easy-to-read note in natural language using the following sections:
1. What Was Discussed (Detailed Overview): (A comprehensive, conversational breakdown of what everyone talked about and the flow of the conversation).
2. Key Takeaways & Agreements: (Rich, informative bullet points detailing key opinions, agreements, and interesting insights).
3. Commitments & To-Do List: (Who promised to do what, with any mentioned details).
"""
        else:
            system_prompt = f"""
You are a professional assistant.
{lang_directive}

{detail_guidance}

{formatting_rules}
"""

    else:
        # Język polski
        lang_directive = "CAŁOŚĆ ODPOWIEDZI MUSI BYĆ WYŁĄCZNIE W JĘZYKU POLSKIM. WSZYSTKIE NAGŁÓWKI, PUNKTY I OPISY MAJĄ BYĆ PO POLSKU."
        
        detail_guidance = (
            "WYCZYPANE I SZCZEGÓŁOWE PODSUMOWANIE DŁUGIEGO NAGRANIA:\n"
            "Przekazane nagranie zawiera obszerną, wielowątkową dyskusję. Przygotuj drobiazgowy, wyczerpujący i bogaty w treść raport:\n"
            "- Bezwzględnie unikaj powierzchownych skrótów myślowych i lakonicznych 1-2 zdań.\n"
            "- Dokładnie opisz każdy poruszony wątek, rozważane warianty, argumenty rozmówców oraz wypracowane kompromisy.\n"
            "- Zachowaj wszystkie konkretne liczby, kwoty, daty, technologie, wskaźniki i nazwiska poruszone w trakcie spotkania.\n"
            "- Rozwiń poszczególne punkty tak, aby osoba nieobecna na spotkaniu poznała pełny kontekst i szczegóły dyskusji."
            if is_long_audio else
            "WYMÓG JAKOŚCI I DOKŁADNOŚCI:\n"
            "Zapewnij wyczerpujące podsumowanie rozmowy. Nie pomijaj żadnych istotnych faktów, argumentów ani ustaleń."
        )

        formatting_rules = (
            "ZASADY FORMATOWANIA:\n"
            "- W punktatorach NIGDY nie otaczaj całych zdań podwójnymi gwiazdkami (np. NIGDY nie pisz: '- **Całe zdanie.**').\n"
            "- Pogrubienie (**) stosuj wyłącznie dla krótkiej etykiety lub imienia osoby na początku punktu (np. '- Jan: opis zadania' lub '- Kwestia budżetu: treść').\n"
            "- Zadbaj o estetyczny, przejrzysty układ tekstu."
        )

        if summary_style == "Biznesowy":
            system_prompt = f"""
Jesteś doświadczonym, profesjonalnym asystentem zarządu i specjalistą od dokumentacji biznesowej.
{lang_directive}

{detail_guidance}

{formatting_rules}

Format wyjściowy musi zawierać 3 rozbudowane sekcje:
1. Cel spotkania i tło problemu: (Wyczerpujący opis powodów zwołania spotkania, kontekstu biznesowego, wyzwań oraz głównych celów).
2. Przebieg dyskusji i kluczowe ustalenia: (Szczegółowe, pogłębione omówienie każdego poruszonego tematu, stanowisk uczestników, analizowanych opcji i podjętych decyzji).
3. Zadania i kolejne kroki (Action Items): (Konkretna, drobiazgowa lista zadań z przypisanymi odpowiedzialnymi osobami, terminami i oczekiwanymi rezultatami. Jeśli nie przypisano zadań, wyraźnie o tym poinformuj).
"""
        elif summary_style == "Naukowy / Akademicki":
            system_prompt = f"""
Jesteś asystentem naukowym i analitykiem badań akademickich.
{lang_directive}

{detail_guidance}

{formatting_rules}

Format wyjściowy musi zawierać 3 rozbudowane sekcje:
1. Główny problem badawczy i kontekst: (Wyczerpujące przedstawienie problematyki badawczej, stawianych pytań, hipotez oraz tła teoretycznego).
2. Pogłębiona argumentacja, koncepcje i metodologia: (Wnikliwa analiza twierdzeń uczestników, przytoczonych dowodów, koncepcji, metodologii oraz kontrargumentów).
3. Wnioski końcowe i perspektywy badawcze: (Syntetyczne, rzetelne podsumowanie wyników dyskusji, otwarte pytania i rekomendacje dalszych kierunków).
"""
        elif summary_style == "Luźna notatka (Casual)":
            system_prompt = f"""
Jesteś bystrym, przyjaznym asystentem.
{lang_directive}

{detail_guidance}

{formatting_rules}

Przygotuj szczegółową, swobodną notatkę w przystępnym języku, podzieloną na sekcje:
1. O czym dokładnie rozmawiano: (Szczegółowy opis poruszonych tematów i dynamiki rozmowy).
2. Najważniejsze wnioski i uzgodnienia: (Rozwinięte, bogate w szczegóły punkty z ustaleń, opinii i spostrzeżeń).
3. Co jest do zrobienia (Następne kroki): (Kto co zadeklarował lub obiecał przygotować).
"""
        else:
            system_prompt = f"""
Jesteś profesjonalnym asystentem.
{lang_directive}

{detail_guidance}

{formatting_rules}
"""

    if custom_instructions and custom_instructions.strip():
        instruction_header = "SPECIAL USER INSTRUCTIONS (STRICT COMPLIANCE MANDATORY)" if is_english else "DODATKOWE INSTRUKCJE OD UŻYTKOWNIKA (BEZWZGLĘDNIE SIĘ DO NICH ZASTOSUJ)"
        system_prompt += f"\n\n--- {instruction_header}: ---\n{custom_instructions.strip()}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Oto transkrypcja:\n\n{transcript_text}"}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.4,
            max_tokens=4096
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
