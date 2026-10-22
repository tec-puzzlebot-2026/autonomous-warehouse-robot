"""
config.py - every tunable setting in one place.

Change WORDS to the 10 commands you want. Every word that is also in the
Google Speech Commands dataset gets thousands of free training recordings
from ~2,000 different speakers (see prepare_speech_commands.py).
Words in Speech Commands v0.02:
  yes no up down left right on off stop go forward backward follow learn
  zero one two three four five six seven eight nine
  bed bird cat dog happy house marvin sheila tree wow visual
Any word NOT in that list you must record yourself with record_for_team.py,
from as many different people as you can. For the current word list only
"stop" exists in Speech Commands, so almost all word data is your own.
Speech Commands is still very useful for the _unknown_ (garbage) class.
"""

# ---- Vocabulary ------------------------------------------------------------
WORDS = ["start", "stop", "pause", "resume", "mapping",
         "restocking", "faster", "slower", "return", "status"]
UNKNOWN = "_unknown_"      # folder with "anything that is NOT a command"

# ---- Paths -----------------------------------------------------------------
# Absolute paths next to this file, so it doesn't matter from which folder you
# run the scripts or the ROS 2 node. realpath() follows the symlinks made by
# `colcon build --symlink-install` back to your source folder.
import os as _os
VOICE_DIR = _os.path.dirname(_os.path.realpath(__file__))
# Data lives in voice/data/ (one level above scripts/). Any layout works as
# long as each clip is inside a folder named after its label, e.g.
#   voice/data/chris/data/stop/chris_nohash_3.wav
#   voice/data/google/_unknown_/0a7c2a8d_nohash_bed0.wav
# train.py searches all sub-folders recursively.
DATA_DIR = _os.environ.get("VOICE_DATA_DIR",
                           _os.path.join(_os.path.dirname(VOICE_DIR), "data"))
GOOGLE_SUBDIR = "google"   # prepare_speech_commands.py writes to data/google/
MODEL_PATH = _os.environ.get("VOICE_MODEL_PATH", _os.path.join(VOICE_DIR, "models.pkl"))

# ---- Audio / features --------------------------------------------------------
SR = 16000                 # sample rate (Hz). 16 kHz is standard for speech.
N_MFCC = 13                # cepstral coefficients per frame
N_MELS = 40                # mel filters
WIN = 400                  # 25 ms analysis window (400 samples @ 16 kHz)
HOP = 160                  # 10 ms hop -> 100 feature frames per second
TRIM_DB = 25               # silence trimming: drop parts 25 dB below peak
MIN_SAMPLES = int(0.15 * SR)

# ---- HMM topology ------------------------------------------------------------
WORD_STATES = 8            # states per word model (left-to-right)
WORD_MIX = 2               # Gaussians per state
GARBAGE_STATES = 6         # "filler" model for non-command sounds (ergodic)
GARBAGE_MIX = 4
N_ITER = 20                # Baum-Welch iterations

# ---- Training data -----------------------------------------------------------
MAX_TRAIN_PER_CLASS = 1000 # cap so training stays a few minutes
                           # (your team's clips are ALWAYS kept; Google fills the rest)
MAX_TEST_PER_CLASS = 300
TEST_FRACTION = 0.15       # fraction of SPEAKERS held out for testing
# With few speakers, choose the test speaker(s) yourself, e.g. ["sofia"].
# Their recordings are used ONLY for testing ("does it work on a new voice?").
# Leave empty to split speakers automatically. For the final model you can set
# it back to [] or USE_ALL_FOR_FINAL = True to train on everyone.
TEST_SPEAKERS = []
USE_ALL_FOR_FINAL = True
DEFAULT_THRESHOLD = -1.0   # used only if there is no unknown test data
TARGET_FALSE_ACCEPT = 0.05 # rejection threshold lets <=5% of non-words through

# ---- Recording (record.py) ---------------------------------------------------
RECORD_SECONDS = 1.5

# ---- Live recognition (recognize.py) -----------------------------------------
BLOCK_SECONDS = 0.02       # 20 ms audio blocks from the microphone
VAD_DB = 12                # speech = louder than noise floor + 12 dB
START_BLOCKS = 3           # 60 ms of speech to trigger
END_BLOCKS = 15            # 300 ms of silence ends the utterance
PRE_ROLL_BLOCKS = 15       # keep 300 ms before trigger (catches soft "s", "f")
MAX_UTTERANCE_SECONDS = 2.5   # "restocking" is a long word
