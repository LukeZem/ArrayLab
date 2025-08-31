"""Legacy demo file updated to reference new song section pipeline.

Use song_sections.process_song_to_doc instead of ad-hoc doc loading.

Example:
    python word.py input.wav output.docx
"""

import argparse
from song_sections import process_song_to_doc


def main():
    parser = argparse.ArgumentParser(description="Process WAV into structured Word doc (intro/verses/chorus/other)")
    parser.add_argument('audio', help='Input WAV file path')
    parser.add_argument('output', help='Output DOCX path')
    parser.add_argument('--model', default='small', help='Whisper model size (tiny, base, small, medium, large)')
    args = parser.parse_args()
    process_song_to_doc(args.audio, args.output, model_size=args.model)


if __name__ == '__main__':
    main()
