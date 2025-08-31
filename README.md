# ArrayLab Song Section Parser

This repo now contains a minimal pipeline to:

1. Transcribe a full-song WAV into time‑stamped lyric segments (offline via Whisper).
2. Heuristically segment those lyrics into: Intro (implied), Verses, Choruses, Other (bridge / outro / misc).
3. Export a well‑formatted Word document listing sections and the full transcript.

## Quick Start

Install dependencies (requires Python 3.9+ and ffmpeg installed on your system):

```bash
pip install -r requirements.txt
```

Ensure `ffmpeg` is available:

```bash
ffmpeg -version
```

Run processing:

```bash
python song_sections.py path/to/song.wav output.docx
```

or the wrapper:

```bash
python word.py path/to/song.wav output.docx
```

## Heuristics Summary

- Segments are grouped into blocks when the silence between consecutive segments exceeds 4 seconds.
- Repeated blocks (Jaccard token similarity >= 0.6) are labeled as Chorus.
- The earliest non-chorus blocks become Verses (blocks similar (>=0.4) to the first verse also become verses).
- Distinct later blocks become Other (Bridge/Outro). The last unmatched block is tagged Outro.
- Intro is inferred from leading silence (>=3s) before first lyrics; otherwise a zero-length intro placeholder is inserted.

Tune thresholds inside `song_sections.py` (constants at top) as needed.

## Extending

Swap out or augment heuristics by replacing `classify_blocks` or layering additional acoustic features (e.g., energy, chord changes) from a music analysis library (librosa, essentia) — not included to keep dependencies light.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| torch / whisper install slow | Use a smaller Whisper model ("tiny" or "base"). |
| ffmpeg not found | Install via system package manager (e.g. `apt-get install ffmpeg`). |
| No chorus detected | Lower `CHORUS_SIMILARITY_THRESHOLD` or verify lyrics actually repeat. |
| Misclassified bridge as verse | Reduce verse similarity threshold (0.4) or add a custom rule. |

## Programmatic Use

```python
from song_sections import parse_song_sections, export_sections_to_docx

sections = parse_song_sections('song.wav', model_size='tiny')
export_sections_to_docx(sections, 'song.docx')
```

## License

Provided as-is; adapt freely.
