import io
import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors


REPORT_TRANSLATIONS = {
    "Polski": {
        "txt_title": "RAPORT ZE SPOTKANIA (AI AUDIO)",
        "txt_date": "Data wygenerowania",
        "txt_summary_header": "--- PODSUMOWANIE SPOTKANIA ---",
        "txt_no_summary": "(Brak wygenerowanego podsumowania)",
        "txt_transcript_header": "--- TRANSKRYPCJA ROZMOWY ---",
        "txt_no_transcript": "(Brak transkrypcji)",
        "pdf_title": "Raport ze Spotkania",
        "pdf_subtitle": "Wygenerowano",
        "pdf_summary_header": "Podsumowanie i wnioski (AI)",
        "pdf_no_summary": "<i>Brak wygenerowanego podsumowania.</i>",
        "pdf_transcript_header": "Zapis rozmowy (Transkrypcja)",
        "pdf_no_transcript": "<i>Brak transkrypcji.</i>",
    },
    "Angielski": {
        "txt_title": "MEETING REPORT (AI AUDIO)",
        "txt_date": "Generated on",
        "txt_summary_header": "--- MEETING SUMMARY ---",
        "txt_no_summary": "(No summary generated)",
        "txt_transcript_header": "--- MEETING TRANSCRIPT ---",
        "txt_no_transcript": "(No transcript available)",
        "pdf_title": "Meeting Report",
        "pdf_subtitle": "Generated on",
        "pdf_summary_header": "Summary & Key Takeaways (AI)",
        "pdf_no_summary": "<i>No summary generated.</i>",
        "pdf_transcript_header": "Meeting Transcript",
        "pdf_no_transcript": "<i>No transcript available.</i>",
    }
}


def _get_labels(language: str) -> dict:
    lang_str = str(language).strip().lower() if language else ""
    if "ang" in lang_str or "en" in lang_str:
        return REPORT_TRANSLATIONS["Angielski"]
    return REPORT_TRANSLATIONS["Polski"]


def clean_txt_summary(summary: str) -> str:
    """
    Usuwa formatowanie markdown (np. **, __, ###) na potrzeby czystego pliku TXT,
    aby w punktatorach nie pojawiały się gwiazdki ** przed i po zdaniach.
    """
    if not summary:
        return ""
    cleaned_lines = []
    for raw_line in summary.splitlines():
        line = re.sub(r'^#{1,6}\s*', '', raw_line)

        prefix = ""
        bullet_match = re.match(r'^([-*•]|\d+\.)\s+(.*)$', line)
        if bullet_match:
            b_type, rest = bullet_match.groups()
            prefix = f"{b_type} "
            line = rest.strip()

        full_bold_match = re.match(r'^\*\*(.*?)\*\*$', line)
        if full_bold_match:
            line = full_bold_match.group(1).strip()
        else:
            line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            line = re.sub(r'__(.*?)__', r'\1', line)

        line = line.replace('**', '').replace('__', '').strip()
        cleaned_lines.append(f"{prefix}{line}" if prefix else line)

    return "\n".join(cleaned_lines)


def format_pdf_line(line: str) -> str:
    """
    Formatuje linię podsumowania dla ReportLab:
    - zamienia markdown ** na tagi <b>...</b>
    - usuwa zbędne gwiazdki ** otaczające całe punkty/zdania
    - obsługuje punktatory i nagłówki
    """
    clean_line = line.strip()
    if not clean_line:
        return ""

    is_heading = False
    if clean_line.startswith("#"):
        clean_line = re.sub(r'^#{1,6}\s*', '', clean_line)
        is_heading = True

    clean_line = clean_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    prefix = ""
    bullet_match = re.match(r'^([-*•]|\d+\.)\s+(.*)$', clean_line)
    if bullet_match:
        b_type, rest = bullet_match.groups()
        prefix = "&bull; " if b_type in ['-', '*', '•'] else f"{b_type} "
        clean_line = rest.strip()

    full_bold_match = re.match(r'^\*\*(.*?)\*\*$', clean_line)
    if full_bold_match:
        inner = full_bold_match.group(1).strip()
        if ":" in inner:
            parts = inner.split(":", 1)
            clean_line = f"<b>{parts[0].strip()}:</b> {parts[1].strip()}"
        else:
            clean_line = inner
    else:
        clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
        clean_line = re.sub(r'__(.*?)__', r'<b>\1</b>', clean_line)

    clean_line = clean_line.replace('**', '').replace('__', '')

    if prefix:
        clean_line = prefix + clean_line
    if is_heading:
        clean_line = f"<b>{clean_line}</b>"

    return clean_line


def _register_polish_font():
    """
    Rejestruje czcionkę obsługującą polskie znaki z katalogu Windows Fonts.
    """
    font_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    
    win_arial = r"C:\Windows\Fonts\arial.ttf"
    win_arial_bd = r"C:\Windows\Fonts\arialbd.ttf"
    
    if os.path.exists(win_arial) and os.path.exists(win_arial_bd):
        try:
            pdfmetrics.registerFont(TTFont("ArialPL", win_arial))
            pdfmetrics.registerFont(TTFont("ArialPL-Bold", win_arial_bd))
            font_name = "ArialPL"
            bold_name = "ArialPL-Bold"
        except Exception as e:
            print(f"Nie udało się załadować czcionki Arial: {e}")
            
    return font_name, bold_name


def generate_txt_report(
    transcript=None, 
    summary=None, 
    export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
    include_timestamps=True,
    speaker_names=None,
    language="Polski"
):
    """
    Generuje zawartość pliku TXT na podstawie wybranych opcji z nagłówkami
    dopasowanymi do wybranego języka (Polski / Angielski) oraz czystym tekstem bez surowych **.
    """
    labels = _get_labels(language)
    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines.append("==========================================")
    lines.append(f"      {labels['txt_title']}      ")
    lines.append(f"      {labels['txt_date']}: {now_str}       ")
    lines.append("==========================================\n")
    
    mode_lower = export_mode.lower()
    has_summary = ("podsumowanie" in mode_lower or "summary" in mode_lower or "pełny" in mode_lower or "full" in mode_lower)
    has_transcript = ("transkrypcja" in mode_lower or "transcript" in mode_lower or "pełny" in mode_lower or "full" in mode_lower)

    if has_summary:
        lines.append(f"{labels['txt_summary_header']}\n")
        if summary:
            cleaned_summary = clean_txt_summary(summary)
            lines.append(cleaned_summary.strip())
        else:
            lines.append(labels['txt_no_summary'])
        lines.append("\n" + "=" * 42 + "\n")
        
    if has_transcript:
        lines.append(f"{labels['txt_transcript_header']}\n")
        if transcript:
            for item in transcript:
                spk = item["speaker"]
                if speaker_names and spk in speaker_names and speaker_names[spk].strip():
                    spk = speaker_names[spk].strip()
                    
                if include_timestamps:
                    lines.append(f"[{item['start_str']} - {item['end_str']}] {spk}: {item['text']}")
                else:
                    lines.append(f"{spk}: {item['text']}")
                lines.append("")
        else:
            lines.append(labels['txt_no_transcript'])
            
    return "\n".join(lines)


def generate_pdf_report(
    transcript=None, 
    summary=None, 
    export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
    include_timestamps=True,
    speaker_names=None,
    language="Polski"
):
    """
    Generuje bufor bajtów z plikiem PDF z nagłówkami w odpowiednim języku (Polski / Angielski)
    oraz poprawnym formatowaniem punktatorów bez surowych **.
    """
    labels = _get_labels(language)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=45, 
        leftMargin=45, 
        topMargin=45, 
        bottomMargin=45
    )
    
    font_regular, font_bold = _register_polish_font()
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=font_bold,
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor("#1f2937")
    )
    subtitle_style = ParagraphStyle(
        'CustomSubTitle',
        parent=styles['Normal'],
        fontName=font_regular,
        fontSize=10,
        textColor=colors.gray,
        alignment=1,
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=14,
        spaceBefore=14,
        spaceAfter=10,
        textColor=colors.HexColor("#1f77b4")
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=font_regular,
        fontSize=10.5,
        leading=15,
        spaceAfter=8
    )
    
    story = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    story.append(Paragraph(labels['pdf_title'], title_style))
    story.append(Paragraph(f"{labels['pdf_subtitle']}: {now_str} | Smart Meeting Summarizer", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=15))
    
    mode_lower = export_mode.lower()
    has_summary = ("podsumowanie" in mode_lower or "summary" in mode_lower or "pełny" in mode_lower or "full" in mode_lower)
    has_transcript = ("transkrypcja" in mode_lower or "transcript" in mode_lower or "pełny" in mode_lower or "full" in mode_lower)

    if has_summary:
        story.append(Paragraph(labels['pdf_summary_header'], heading_style))
        if summary:
            for raw_paragraph in summary.splitlines():
                formatted_p = format_pdf_line(raw_paragraph)
                if formatted_p:
                    story.append(Paragraph(formatted_p, body_style))
        else:
            story.append(Paragraph(labels['pdf_no_summary'], body_style))
        story.append(Spacer(1, 15))
        
    if has_transcript:
        story.append(Paragraph(labels['pdf_transcript_header'], heading_style))
        if transcript:
            for item in transcript:
                spk = item["speaker"]
                if speaker_names and spk in speaker_names and speaker_names[spk].strip():
                    spk = speaker_names[spk].strip()
                    
                safe_text = item["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe_spk = spk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                
                if include_timestamps:
                    line_html = f"<font name='{font_bold}' color='#555555'>[{item['start_str']} - {item['end_str']}]</font> <font name='{font_bold}'>{safe_spk}:</font> {safe_text}"
                else:
                    line_html = f"<font name='{font_bold}'>{safe_spk}:</font> {safe_text}"
                    
                story.append(Paragraph(line_html, body_style))
        else:
            story.append(Paragraph(labels['pdf_no_transcript'], body_style))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
