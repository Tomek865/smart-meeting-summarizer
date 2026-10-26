from transcription import transcribe_full_audio

def format_time(seconds):
    """Pomocnicza funkcja do formatowania czasu w sekundach na MM:SS."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

def build_synchronized_transcript(trans_path, diarization_results, whisper_model, custom_vocabulary="", max_gap=2.0):
    """
    Ciągła synchronizacja audio (Late Fusion):
    1. Krok A: Jedno wywołanie modelu Whisper dla całego pliku trans_path.
    2. Krok B: Algorytm Time Intersection - dla każdego segmentu tekstu znajduje przedział
       czasowy mówcy z diarization_results o największym overlapie.
    3. Krok C: Łączenie sąsiadujących wypowiedzi tego samego mówcy (jeśli przerwa jest mała).
    Zwraca listę gotową dla UI: [{"start_str": "MM:SS", "end_str": "MM:SS", "speaker": str, "text": str}].
    """
    whisper_segments = transcribe_full_audio(trans_path, whisper_model, custom_vocabulary=custom_vocabulary)

    if not whisper_segments:
        return []

    assigned_segments = []
    for w_seg in whisper_segments:
        w_start = w_seg["start"]
        w_end = w_seg["end"]
        w_text = w_seg["text"].strip()

        if not w_text:
            continue

        best_speaker = None
        max_overlap = 0.0

        for d_seg in diarization_results:
            overlap = max(0.0, min(w_end, d_seg["end"]) - max(w_start, d_seg["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg["speaker"]

        if best_speaker is None:
            if diarization_results:
                closest_seg = min(
                    diarization_results,
                    key=lambda d: max(0.0, d["start"] - w_end, w_start - d["end"])
                )
                best_speaker = closest_seg["speaker"]
            else:
                best_speaker = "SPEAKER_00"

        assigned_segments.append({
            "start": w_start,
            "end": w_end,
            "speaker": best_speaker,
            "text": w_text
        })

    if not assigned_segments:
        return []

    merged_blocks = []
    for seg in assigned_segments:
        if not merged_blocks:
            merged_blocks.append(dict(seg))
            continue

        prev = merged_blocks[-1]
        gap = seg["start"] - prev["end"]

        if seg["speaker"] == prev["speaker"] and gap <= max_gap:
            prev["end"] = max(prev["end"], seg["end"])
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
        else:
            merged_blocks.append(dict(seg))

    ui_transcript = []
    for block in merged_blocks:
        ui_transcript.append({
            "start_str": format_time(block["start"]),
            "end_str": format_time(block["end"]),
            "speaker": block["speaker"],
            "text": block["text"]
        })

    return ui_transcript
