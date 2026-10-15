import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors


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
    speaker_names=None
):
    """
    Generuje zawartość pliku TXT na podstawie wybranych opcji.
    """
    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines.append("==========================================")
    lines.append("      RAPORT ZE SPOTKANIA (AI AUDIO)      ")
    lines.append(f"      Data wygenerowania: {now_str}       ")
    lines.append("==========================================\n")
    
    if export_mode in ["Tylko podsumowanie", "Pełny raport (Podsumowanie + Transkrypcja)"]:
        lines.append("--- PODSUMOWANIE SPOTKANIA ---\n")
        if summary:
            lines.append(summary.strip())
        else:
            lines.append("(Brak wygenerowanego podsumowania)")
        lines.append("\n" + "=" * 42 + "\n")
        
    if export_mode in ["Tylko transkrypcja", "Pełny raport (Podsumowanie + Transkrypcja)"]:
        lines.append("--- TRANSKRYPCJA ROZMOWY ---\n")
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
            lines.append("(Brak transkrypcji)")
            
    return "\n".join(lines)


def generate_pdf_report(
    transcript=None, 
    summary=None, 
    export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
    include_timestamps=True,
    speaker_names=None
):
    """
    Generuje bufor bajtów z plikiem PDF (z pełną obsługą polskich znaków).
    """
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
    
    story.append(Paragraph("Raport ze Spotkania", title_style))
    story.append(Paragraph(f"Wygenerowano: {now_str} | Smart Meeting Summarizer", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=15))
    
    if export_mode in ["Tylko podsumowanie", "Pełny raport (Podsumowanie + Transkrypcja)"]:
        story.append(Paragraph("Podsumowanie i wnioski (AI)", heading_style))
        if summary:
            clean_summary = summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            for paragraph in clean_summary.split("\n"):
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), body_style))
        else:
            story.append(Paragraph("<i>Brak wygenerowanego podsumowania.</i>", body_style))
        story.append(Spacer(1, 15))
        
    if export_mode in ["Tylko transkrypcja", "Pełny raport (Podsumowanie + Transkrypcja)"]:
        story.append(Paragraph("Zapis rozmowy (Transkrypcja)", heading_style))
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
            story.append(Paragraph("<i>Brak transkrypcji.</i>", body_style))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()