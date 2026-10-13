def merge_transcription_and_diarization(transcription_data, diarization_data):
    """
    Łączy segmenty tekstu z odpowiadającymi im mówcami na podstawie 
    największego pokrycia w czasie (overlap).
    """
    combined_results = []
    
    for t_seg in transcription_data:
        t_start = t_seg['start']
        t_end = t_seg['end']
        
        speaker_overlaps = {}
        
        for d_seg in diarization_data:
            d_start = d_seg['start']
            d_end = d_seg['end']
            
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                speaker = d_seg['speaker']
                if speaker not in speaker_overlaps:
                    speaker_overlaps[speaker] = 0
                speaker_overlaps[speaker] += overlap_duration
        
        if speaker_overlaps:
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
        else:
            best_speaker = "NIEZNANY"
            
        combined_results.append({
            "start_str": t_seg['start_str'],
            "end_str": t_seg['end_str'],
            "speaker": best_speaker,
            "text": t_seg['text']
        })
        
    merged_script = []
    for current in combined_results:
        if not merged_script:
            merged_script.append(current)
            continue
            
        previous = merged_script[-1]
        if previous['speaker'] == current['speaker']:
            previous['end_str'] = current['end_str']
            previous['text'] += " " + current['text']
        else:
            merged_script.append(current)
            
    return merged_script