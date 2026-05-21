"""
Wrapper that replaces torchaudio.save with soundfile to bypass the
torchcodec/FFmpeg DLL issue on Windows, then runs demucs normally.
"""
import sys
import soundfile as sf
import torchaudio


def _soundfile_save(uri, src, sample_rate, **kwargs):
    # src is a torch tensor shaped (channels, samples); soundfile wants (samples, channels)
    audio_np = src.numpy().T
    sf.write(str(uri), audio_np, sample_rate)


torchaudio.save = _soundfile_save

from demucs.__main__ import main
sys.exit(main())
