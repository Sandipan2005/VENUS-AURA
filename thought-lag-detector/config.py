
import os

class Config:
	SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")
	SQLALCHEMY_DATABASE_URI = os.environ.get(
		"DATABASE_URL", "sqlite:///thoughtlag.db"
	)
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
	MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
	# Whisper model (Hugging Face free model via faster-whisper)
	ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "tiny.en")  # or "openai/whisper-tiny.en"
	# VAD params
	VAD_MODE = int(os.environ.get("VAD_MODE", 2))  # 0-3 aggressiveness
	# Baseline
	BASELINE_MIN_PROMPTS = int(os.environ.get("BASELINE_MIN_PROMPTS", 8))
	# Fast mode: skip heavier features (pitch) for lower latency
	FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"
	# Privacy / retention flags
	RETAIN_AUDIO = os.environ.get("RETAIN_AUDIO", "1") == "1"  # if false, delete wav after feature extraction
	STORE_TRANSCRIPTS = os.environ.get("STORE_TRANSCRIPTS", "1") == "1"  # if false, discard transcript text
	ALLOW_EXPORT = os.environ.get("ALLOW_EXPORT", "1") == "1"
	# Provisional scoring interval seconds
	PROVISIONAL_INTERVAL = float(os.environ.get("PROVISIONAL_INTERVAL", 1.0))
	# Calibration durations (seconds)
	CALIB_SILENCE_SEC = float(os.environ.get("CALIB_SILENCE_SEC", 3.0))
	CALIB_VOWEL_SEC = float(os.environ.get("CALIB_VOWEL_SEC", 3.0))
	# Advanced dual-stage VAD
	ADV_VAD_ENABLED = os.environ.get("ADV_VAD_ENABLED", "0") == "1"
	VAD_FUSION_WEIGHTS = {
		"webrtc": float(os.environ.get("VAD_W_WEBRTC", 0.4)),
		"ml": float(os.environ.get("VAD_W_ML", 0.4)),
		"energy": float(os.environ.get("VAD_W_ENERGY", 0.2)),
	}
	VAD_HANGOVER_FRAMES = int(os.environ.get("VAD_HANGOVER_FRAMES", 6))  # keep speech true this many frames after last positive
	VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", 0.5))
	# Whisper streaming / refinement
	WHISPER_PARTIAL_WINDOW_SEC = float(os.environ.get("WHISPER_PARTIAL_WINDOW_SEC", 6.0))
	WHISPER_PARTIAL_INTERVAL_SEC = float(os.environ.get("WHISPER_PARTIAL_INTERVAL_SEC", 1.0))
	WHISPER_PARTIAL_MAX_TOKENS = int(os.environ.get("WHISPER_PARTIAL_MAX_TOKENS", 64))
	WHISPER_REFINE_MODEL = os.environ.get("WHISPER_REFINE_MODEL", "small")  # 'none' to disable
	WHISPER_REFINE_MIN_SEC = float(os.environ.get("WHISPER_REFINE_MIN_SEC", 2.5))
	WHISPER_REFINE_ENABLED = WHISPER_REFINE_MODEL.lower() != 'none'
