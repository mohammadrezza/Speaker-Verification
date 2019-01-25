import scipy.io.wavfile as wav
from utility import *
from proj_paths import *
from DTW.extract_features import extract

if __name__ == "__main__":
    for raw_voice, joined_voice in collect_files(REF_VOICES_PATH):
        _, sig = wav.read(joined_voice)
        feats = extract(sig)
        save(feats, raw_voice, DTW_MODELS_PATH)
