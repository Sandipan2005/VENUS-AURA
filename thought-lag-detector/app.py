
import os
import uuid
import time
import csv
import base64
import math
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from config import Config
from models.asr import ASR
from models.features import (
    analyze_wav,
    refine_reaction_with_vad,
)
from models.scoring import ScoreModel
from models.ml_model import MLModel
from models.stress_model import StressModel, compute_frame_features
from models.storage import db, Session, PromptResult, ensure_dirs
from sqlalchemy import func
from flask_socketio import SocketIO, join_room # type: ignore

app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
CORS(app)
db.init_app(app)

# Socket.IO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Create folders and DB
with app.app_context():
    ensure_dirs(app)
    db.create_all()

# Load models at startup
asr = ASR(app.config["ASR_MODEL_NAME"])
scorer = ScoreModel()
ml_model = MLModel()
stress_model = StressModel()

# Live streaming buffers (per client) for lightweight real-time previews
live_buffers: dict[str, bytearray] = {}
last_partial_decode: dict[str, float] = {}
calibration_state: dict[str, dict] = {}  # client_id -> {noise_floor, base_pitch}
last_provisional_emit: dict[str, float] = {}
vad_states: dict[str, dict] = {}  # per-client advanced VAD rolling state
vad_stats: dict[str, dict] = {}   # metrics for /api/vad_stats

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # fallback if missing, live metrics will degrade

# Optional prewarm for ASR (loads model weights fully)
try:
    asr.transcribe(os.path.join(app.config["UPLOAD_FOLDER"], "_dummy_missing.wav"))
except Exception:
    pass


# Socket handlers
@socketio.on("register")
def handle_register(data):
    cid = data.get("client_id") if isinstance(data, dict) else None
    role = (data.get("role") if isinstance(data, dict) else None) or "client"
    if cid:
        join_room(cid)
    if role == "dashboard":
        join_room("dashboard")
    _write_socket_log(f"register:{cid}:{role}")


def _write_socket_log(line: str):
    try:
        log_dir = os.path.join(app.instance_path)
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "socket_log.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} - {line}\n")
    except Exception:
        pass


@socketio.on("ack")
def handle_ack(payload):
    try:
        pr_id = payload.get("pr_id")
        cid = payload.get("client_id")
        _write_socket_log(f"ack from {cid} pr={pr_id}")
        # confirm back to client
        socketio.emit("ack_confirm", {"pr_id": pr_id, "client_id": cid}, room=cid)  # type: ignore
    except Exception:
        pass


@socketio.on("diagnostic")
def handle_diagnostic(payload):
    try:
        _write_socket_log(f"diag:{payload}")
    except Exception:
        pass

# Demo prompt set (mix easy->hard)
PROMPTS = [
    "What day is it today?",
    "Name as many fruits as you can in ten seconds.",
    "What is twelve plus seventeen?",
    "Spell the word 'catalog' backward.",
    "Say the colors you see, not the words: red, blue, green.",
    "Name three animals that live in the ocean.",
    "What is twenty-three times four?",
    "Repeat: nine, one, five, two, six.",
    "Describe your favorite hobby in one sentence.",
    "What month comes before September?",
]

@app.route("/")
def index():
    return render_template("index.html", prompts=PROMPTS)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """Receive a small PCM16 base64 chunk from client and emit lightweight live metrics.

    Expects: { client_id, pcm16, sample_rate, prompt_idx }
    Emits: live_update -> same room with { rms, duration, prompt_idx }
    """
    try:
        cid = data.get("client_id") if isinstance(data, dict) else None
        b64 = data.get("pcm16") if isinstance(data, dict) else None
        sr = int(data.get("sample_rate") or 48000)
        pidx = data.get("prompt_idx")
        if not cid or not b64:
            return
        raw = base64.b64decode(b64)
        buf = live_buffers.setdefault(cid, bytearray())
        buf.extend(raw)
    # Compute RMS on this chunk only for responsiveness
        if np is not None and len(raw) >= 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if arr.size:
                rms = float(math.sqrt(float((arr * arr).mean())))
            else:
                rms = 0.0
        else:
            # crude fallback
            rms = 0.0
            for i in range(0, len(raw), 2):
                sample = int.from_bytes(raw[i:i+2], 'little', signed=True) / 32768.0
                rms += sample * sample
            if len(raw) >= 2:
                rms = math.sqrt(rms / (len(raw)/2))
        duration = len(buf) / (2 * sr)
        # Advanced VAD fusion (optional)
        speech_prob = None
        stable = None
        if app.config.get("ADV_VAD_ENABLED") and np is not None:
            st = vad_states.setdefault(cid, {
                "frames": [],
                "hang": 0,
                "last_label": 0,
            })
            # Extract quick DSP features
            # Reconstruct numpy array for this fusion stage (small performance cost acceptable)
            arr_adv = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            zcr = float((np.abs(np.diff(np.sign(arr_adv))) > 0).mean()) if arr_adv.size > 1 else 0.0
            energy = rms
            flatness = 0.0
            try:
                import numpy.fft as _fft
                if arr_adv.size > 64:
                    spec = np.abs(_fft.rfft(arr_adv * np.hanning(arr_adv.size))) + 1e-9
                    geo = np.exp(np.mean(np.log(spec)))
                    arith = np.mean(spec)
                    flatness = float(geo / arith)
            except Exception:
                pass
            # Placeholder ML probability (replace with silero invocation if installed)
            ml_prob = 0.0
            if 'silero_vad' in globals():  # type: ignore
                try:
                    global silero_vad
                    ml_prob = float(silero_vad(arr_adv, sample_rate=sr).item())  # type: ignore
                except Exception:
                    ml_prob = 0.0
            else:
                # heuristic pseudo-ML
                ml_prob = float(min(1.0, max(0.0, (energy*8) * (1 - flatness*0.5))))
            # webrtc decision approximated by energy crossing (since we only have raw chunk), fallback on ml_prob
            webrtc_like = 1.0 if energy > 0.02 else 0.0
            w = app.config.get("VAD_FUSION_WEIGHTS", {"webrtc":0.4,"ml":0.4,"energy":0.2})
            speech_prob = float(
                w.get("webrtc",0.4)*webrtc_like +
                w.get("ml",0.4)*ml_prob +
                w.get("energy",0.2)*min(1.0, energy*10)
            )
            threshold = app.config.get("VAD_THRESHOLD", 0.5)
            label = 1 if speech_prob >= threshold else 0
            if label == 0 and st.get("hang",0) > 0:
                label = 1
                st["hang"] -= 1
            elif label == 1:
                st["hang"] = app.config.get("VAD_HANGOVER_FRAMES",6)
            # stability: require 2 consecutive same labels
            frames = st.setdefault("frames", [])
            frames.append(label)
            if len(frames) > 4:
                frames.pop(0)
            stable = (len(frames) >= 2 and frames[-1] == frames[-2])
            # stats
            vs = vad_stats.setdefault(cid, {"false_starts":0,"frames":0,"speech_frames":0})
            vs["frames"] += 1
            if label == 1:
                vs["speech_frames"] += 1
            if label == 0 and st.get("last_label",0)==1 and speech_prob < (threshold*0.5):
                vs["false_starts"] += 1
            st["last_label"] = label
        payload = {"client_id": cid, "rms": rms, "duration": duration, "prompt_idx": pidx}
        if speech_prob is not None:
            payload["speech_prob"] = speech_prob
            payload["speech_stable"] = stable
        # Throttled partial ASR every configured interval after at least 0.9s audio
        now = time.time()
        interval_p = app.config.get("WHISPER_PARTIAL_INTERVAL_SEC", 1.0)
        need_decode = duration > 0.9 and (now - last_partial_decode.get(cid, 0)) > interval_p
        if need_decode:
            try:
                max_window_sec = app.config.get("WHISPER_PARTIAL_WINDOW_SEC", 6.0)
                bytes_per_sec = 2 * sr
                if len(buf) > bytes_per_sec * max_window_sec:
                    start = len(buf) - int(bytes_per_sec * max_window_sec)
                    live_buffers[cid] = live_buffers[cid][start:]
                tmp_name = f"_live_{cid}.wav"
                tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                import wave  # local import
                with wave.open(tmp_path, 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(live_buffers[cid])
                partial = asr.partial_transcribe(tmp_path, max_tokens=app.config.get("WHISPER_PARTIAL_MAX_TOKENS",64))
                payload["partial_transcript"] = partial.get("text", "").strip()
                last_partial_decode[cid] = now
            except Exception as e:  # pragma: no cover
                _write_socket_log(f"partial_err:{e}")
        socketio.emit("live_update", payload, to=cid)  # type: ignore
        # Provisional scoring (lightweight) every PROVISIONAL_INTERVAL seconds using current buffer
        try:
            interval = app.config.get("PROVISIONAL_INTERVAL", 1.0)
            now2 = time.time()
            if np is not None and (now2 - last_provisional_emit.get(cid, 0)) >= interval and len(buf) > sr * 0.6:
                arr_full = np.frombuffer(buf, dtype=np.uint8)  # reinterpret after conversion? we'll rebuild from int16
                # For efficiency use only last 3 seconds
                bytes_per_sec = 2 * sr
                data_bytes = bytes(buf[-int(bytes_per_sec*3):]) if len(buf) > bytes_per_sec*3 else bytes(buf)
                arr_i16 = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Simple provisional features: rms var proxy, zero-crossing for activity, length for speaking duration
                if arr_i16.size > 0:
                    rms_series = arr_i16.reshape(-1, max(1, int(0.02*sr)))  # rough framing
                    rms_vals = np.sqrt((rms_series*rms_series).mean(axis=1)) if rms_series.ndim > 1 else np.array([rms])
                    intensity_var = float(np.var(rms_vals))
                    speaking_activity = float(np.mean(rms_vals > (0.5 * (np.mean(rms_vals) + 1e-6))))
                else:
                    intensity_var = 0.0
                    speaking_activity = 0.0
                provisional_feats = {
                    "reaction_ms_server": 0.0,  # unknown until end
                    "pause_ratio": 1.0 - speaking_activity,
                    "pitch_cv": 0.0,
                    "intensity_var": intensity_var,
                    "speaking_rate_wps": 0.0,
                    "filler_count": 0,
                    "jitter_local": 0.0,
                    "shimmer_local": 0.0,
                    "breath_hf_ratio": 0.0,
                }
                scores = scorer.score(cid, provisional_feats)
                socketio.emit("provisional_score", {"client_id": cid, "scores": scores, "provisional": True, "prompt_idx": pidx}, to=cid)  # type: ignore
                last_provisional_emit[cid] = now2
        except Exception as e:  # pragma: no cover
            _write_socket_log(f"prov_err:{e}")
    except Exception as e:  # pragma: no cover
        _write_socket_log(f"live_err:{e}")

@app.route("/api/next_prompt", methods=["GET"])
def next_prompt():
    idx = int(request.args.get("idx", 0))
    if idx < 0 or idx >= len(PROMPTS):
        return jsonify({"done": True})
    return jsonify({"done": False, "prompt": PROMPTS[idx], "idx": idx})

@app.route("/api/session/start", methods=["POST"])
def start_session():
    data = request.get_json(force=True)
    client_id = data.get("client_id") or str(uuid.uuid4())
    is_baseline = bool(data.get("baseline", False))
    s = Session(client_id=client_id, started_at=datetime.utcnow(), is_baseline =is_baseline) # type: ignore
    db.session.add(s)
    db.session.commit()
    return jsonify({"session_id": s.id, "client_id": client_id})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    # multipart/form-data with fields and audio wav
    t_process_start = time.time()
    client_id = request.form.get("client_id")
    session_id_str = request.form.get("session_id")
    if session_id_str is None:
        return jsonify({"error": "session_id is required"}), 400
    session_id = int(session_id_str)
    prompt_text = request.form.get("prompt_text")
    idx = int(request.form.get("prompt_idx", -1))
    client_seq_str = request.form.get("client_seq")
    client_seq = int(client_seq_str) if client_seq_str is not None else idx
    t_prompt_end_ms_str = request.form.get("t_prompt_end_ms")
    if t_prompt_end_ms_str is None:
        return jsonify({"error": "t_prompt_end_ms is required"}), 400
    t_prompt_end_ms = float(t_prompt_end_ms_str)
    t_onset_ms_str = request.form.get("t_onset_ms")
    if t_onset_ms_str is None:
        return jsonify({"error": "t_onset_ms is required"}), 400
    t_onset_ms_client = float(t_onset_ms_str)
    reaction_ms_client_str = request.form.get("reaction_ms_client")
    if reaction_ms_client_str is None:
        return jsonify({"error": "reaction_ms_client is required"}), 400
    reaction_ms_client = float(reaction_ms_client_str)
    sample_rate_str = request.form.get("sample_rate")
    if sample_rate_str is None:
        return jsonify({"error": "sample_rate is required"}), 400
    sample_rate_client = int(sample_rate_str)

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "no audio uploaded"}), 400

    filename = audio.filename if audio.filename is not None else "audio.wav"
    fn = f"{uuid.uuid4().hex}_{secure_filename(filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
    audio.save(save_path)

    # Server-side analysis
    fast_override = request.form.get("fast") == "1"
    fast_override = request.form.get("fast") == "1"
    fast_mode_used = (fast_override or app.config.get("FAST_MODE", False))
    feat = analyze_wav(
        save_path,
        vad_mode=app.config["VAD_MODE"],
        fast_mode=fast_mode_used
    )
    # If calibration available, adjust pitch-based features (placeholder: could subtract baseline pitch mean)
    calib = calibration_state.get(client_id or "")
    if calib and feat.get("pitch_mean_hz") and calib.get("base_pitch"):
        try:
            feat["pitch_mean_hz_calib"] = float(feat["pitch_mean_hz"]) - float(calib["base_pitch"])
        except Exception:
            pass

    # Refine reaction with server VAD
    reaction_ms_server = refine_reaction_with_vad(
        feat.get("vad_segments"),
        t_prompt_end_ms=t_prompt_end_ms
    ) or reaction_ms_client

    # ASR + fillers + speaking rate
    asr_result = asr.transcribe(save_path)
    transcript = asr_result["text"]
    words = asr_result.get("words", [])
    filler_words = {"um", "uh", "er", "erm", "hmm", "mm"}
    filler_count = sum(1 for w in words if w["word"].lower().strip(".,?!") in filler_words)

    # Speaking rate (words/sec) using ASR timestamps when available
    speech_dur = feat.get("speech_duration_sec", 1e-6)
    wps = max(len(words) / max(speech_dur, 1e-3), 0.0)

    # Merge features
    features = {
        "reaction_ms_client": reaction_ms_client,
        "reaction_ms_server": reaction_ms_server,
        "pause_ratio": feat.get("pause_ratio"),
        "mean_pause_ms": feat.get("mean_pause_ms"),
        "pitch_mean_hz": feat.get("pitch_mean_hz"),
        "pitch_cv": feat.get("pitch_cv"),
        "intensity_var": feat.get("intensity_var"),
    "jitter_local": feat.get("jitter_local"),
    "shimmer_local": feat.get("shimmer_local"),
    "breath_hf_ratio": feat.get("breath_hf_ratio"),
        "speaking_rate_wps": wps,
        "filler_count": filler_count,
        "duration_sec": feat.get("duration_sec"),
        "speech_duration_sec": speech_dur,
        "transcript": transcript,
    }

    # Neural stress model inference (optional torch)
    model_stress_prob = None
    try:
        if stress_model.enabled and speech_dur >= 0.4:
            import soundfile as sf
            import numpy as _np
            y, sr_loaded = sf.read(save_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            feats_frame = compute_frame_features(y, sr=sr_loaded)
            sm_out = stress_model.infer(feats_frame)
            model_stress_prob = sm_out.get("model_stress_prob")
            if sm_out.get("embedding"):
                features["embedding_mean"] = sm_out["embedding"]  # stored for potential adaptation
    except Exception:
        pass

    # Heuristic baseline-based scores
    scores = scorer.score(client_id, features)
    if model_stress_prob is not None:
        # Simple fusion: average for now
        fused = 0.5 * scores["stress"] + 50.0 * model_stress_prob
        scores["stress_fused"] = float(fused)
        scores["model_stress_prob"] = float(model_stress_prob)
    # Online ML augmentation (pseudo-labeled by heuristic stress)
    try:
        scores.update(ml_model.update_and_predict(features, scores["stress"]))
    except Exception:
        pass

    # Baseline aggregates for this client (across baseline sessions)
    basestress = basefocus = None
    try:
        basestress, basefocus = db.session.query(
            func.avg(PromptResult.stress_score),
            func.avg(PromptResult.focus_score)
        ).join(Session, Session.id == PromptResult.session_id).filter(
            Session.client_id == client_id,
            Session.is_baseline.is_(True)
        ).one()
    except Exception:
        pass
    baseline_obj = None
    if basestress is not None and basefocus is not None and basestress is not None:
        baseline_obj = {
            "stress_mean": float(basestress),
            "focus_mean": float(basefocus),
            "stress_delta": float(scores["stress"] - basestress),
            "focus_delta": float(scores["focus"] - basefocus),
        }

    # Persist
    pr = PromptResult(
        session_id=session_id,
        prompt_idx=idx,
        prompt_text=prompt_text,
        audio_path=save_path,
        features=features,
        stress_score=scores["stress"],
        focus_score=scores["focus"],
        created_at=datetime.utcnow()
    )
    db.session.add(pr)
    db.session.commit()

    # Respect privacy flags: delete wav if retention disabled
    if not app.config.get("RETAIN_AUDIO", True):
        try:
            os.remove(save_path)
            pr.audio_path = None  # type: ignore
            db.session.commit()
        except Exception:
            pass

    processing_ms = int((time.time() - t_process_start) * 1000)

    # Optional refinement scheduling (non-blocking)
    try:
        if app.config.get("WHISPER_REFINE_ENABLED") and feat.get("speech_duration_sec",0) >= app.config.get("WHISPER_REFINE_MIN_SEC",2.5):
            refine_model = app.config.get("WHISPER_REFINE_MODEL","small")
            # simple background thread
            import threading
            def _refine_job(path=save_path, pr_id=pr.id, cid=client_id, model_name=refine_model):
                try:
                    refined = asr.refine_transcribe(path, model_name=model_name)
                    socketio.emit("transcript_update", {"pr_id": pr_id, "client_id": cid, "refined_transcript": refined.get("text")}, room=cid)  # type: ignore
                except Exception as e:  # pragma: no cover
                    _write_socket_log(f"refine_err:{e}")
            threading.Thread(target=_refine_job, daemon=True).start()
    except Exception:
        pass

    # Emit real-time update to the client room if connected
    try:
        payload = {
            "pr_id": pr.id,
            "prompt_idx": idx,
            "client_seq": client_seq,
            "features": features,
            "scores": scores,
            "transcript": transcript,
            "processing_ms": processing_ms,
            "fast_mode": fast_mode_used,
        }
        if baseline_obj:
            payload["baseline"] = baseline_obj
        socketio.emit("analysis_result", payload, room=client_id)  # type: ignore
        socketio.emit("analysis_result", payload, room="dashboard")  # type: ignore
        _write_socket_log(f"emit pr={pr.id} cid={client_id} idx={idx} base={'y' if baseline_obj else 'n'}")
    except Exception:
        pass

    resp_payload = {
        "features": features,
        "scores": scores,
        "transcript": transcript,
        "pr_id": pr.id,
        "prompt_idx": idx,
        "client_seq": client_seq,
        "processing_ms": processing_ms,
        "debug": {
            "server_onset_ms": reaction_ms_server,
            "client_onset_ms": t_onset_ms_client,
            "t_prompt_end_ms": t_prompt_end_ms,
        },
    }
    if baseline_obj:
        resp_payload["baseline"] = baseline_obj
    resp_payload["fast_mode"] = fast_mode_used
    return jsonify(resp_payload)

@app.route("/api/summary", methods=["GET"])
def summary():
    client_id = request.args.get("client_id")
    session_id = request.args.get("session_id")
    q = PromptResult.query
    if session_id:
        q = q.filter_by(session_id=int(session_id))
    if client_id:
        q = q.join(Session, Session.id == PromptResult.session_id).filter(Session.client_id == client_id)
    results = q.order_by(PromptResult.created_at.asc()).all() # type: ignore
    series = []
    for r in results:
        series.append({
            "ts": r.created_at.isoformat(),
            "idx": r.prompt_idx,
            "stress": r.stress_score,
            "focus": r.focus_score,
            "reaction_ms": r.features.get("reaction_ms_server", r.features.get("reaction_ms_client")),
            "pause_ratio": r.features.get("pause_ratio"),
        })
    return jsonify({"series": series})

@app.route('/api/vad_stats', methods=['GET'])
def get_vad_stats():
    cid = request.args.get('client_id')
    if cid and cid in vad_stats:
        vs = vad_stats[cid]
        ratio = (vs.get('speech_frames',0) / vs.get('frames',1)) if vs else 0
        return jsonify({"client_id": cid, **vs, "speech_ratio": ratio})
    return jsonify({"clients": list(vad_stats.keys())})

@app.route("/api/calibrate/start", methods=["POST"])
def calibrate_start():
    data = request.get_json(force=True)
    cid = data.get("client_id")
    if not cid:
        return jsonify({"error": "client_id required"}), 400
    calibration_state[cid] = {"noise_samples": [], "pitch_samples": []}
    return jsonify({"status": "started"})

@app.route("/api/calibrate/push", methods=["POST"])
def calibrate_push():
    # Accept small PCM16 base64 chunk with phase label: noise or vowel
    d = request.get_json(force=True)
    cid = d.get("client_id")
    phase = d.get("phase")  # 'silence' | 'vowel'
    b64 = d.get("pcm16")
    sr = int(d.get("sample_rate") or 16000)
    if not cid or not b64 or phase not in ("silence", "vowel"):
        return jsonify({"error": "bad request"}), 400
    if cid not in calibration_state:
        return jsonify({"error": "calibration not started"}), 400
    raw = base64.b64decode(b64)
    if np is not None:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if phase == "silence":
            calibration_state[cid]["noise_samples"].append(float(np.sqrt((arr*arr).mean())))
        else:
            # rudimentary pitch estimate via autocorr center clipping
            try:
                if arr.size > 200:
                    f0 = _quick_pitch(arr, sr)
                    if f0:
                        calibration_state[cid]["pitch_samples"].append(f0)
            except Exception:
                pass
    return jsonify({"ok": True})

@app.route("/api/calibrate/finish", methods=["POST"])
def calibrate_finish():
    data = request.get_json(force=True)
    cid = data.get("client_id")
    if not cid or cid not in calibration_state:
        return jsonify({"error": "not started"}), 400
    st = calibration_state[cid]
    noise_floor = float(np.median(st.get("noise_samples", [0.0]))) if np is not None else 0.0
    base_pitch = float(np.median(st.get("pitch_samples", [0.0]))) if np is not None else 0.0
    calibration_state[cid] = {"noise_floor": noise_floor, "base_pitch": base_pitch, "ts": time.time()}
    return jsonify({"noise_floor": noise_floor, "base_pitch": base_pitch})

def _quick_pitch(y, sr):  # simplistic autocorrelation-based
    try:
        import numpy as _np
        y = y - _np.mean(y)
        if _np.max(_np.abs(y)) > 0:
            y = y / (1e-9 + _np.max(_np.abs(y)))
        corr = _np.correlate(y, y, mode='full')
        corr = corr[corr.size//2:]
        # ignore first small lags
        min_lag = int(sr/500)
        max_lag = int(sr/60)
        seg = corr[min_lag:max_lag]
        if seg.size == 0:
            return None
        lag = int(min_lag + seg.argmax())
        f0 = sr / max(lag,1)
        if 60 < f0 < 500:
            return float(f0)
    except Exception:
        return None
    return None

@app.route("/api/session/<int:session_id>/purge", methods=["POST"])
def purge_session(session_id: int):
    if not app.config.get("ALLOW_EXPORT", True):
        return jsonify({"error": "disabled"}), 403
    res = PromptResult.query.filter_by(session_id=session_id).all()  # type: ignore
    removed = 0
    for r in res:
        if r.audio_path and os.path.exists(r.audio_path):
            try:
                os.remove(r.audio_path)
                r.audio_path = None  # type: ignore
                removed += 1
            except Exception:
                pass
    db.session.commit()
    return jsonify({"purged_audio": removed})

@app.route("/api/session/<int:session_id>/export.csv", methods=["GET"])
def export_session_csv(session_id: int):
    results = (
        PromptResult.query.filter_by(session_id=session_id)
        .order_by(PromptResult.prompt_idx)
        .all()  # type: ignore
    )
    if not results:
        return jsonify({"error": "session not found or empty"}), 404
    output = StringIO()
    w = csv.writer(output)
    w.writerow([
        "prompt_idx","prompt_text","stress_score","focus_score","reaction_ms_server","reaction_ms_client","pause_ratio","speaking_rate_wps","filler_count","pitch_mean_hz","created_at"
    ])
    for r in results:
        f = r.features or {}
        w.writerow([
            r.prompt_idx,
            (r.prompt_text or '').replace('\n',' '),
            r.stress_score,
            r.focus_score,
            f.get("reaction_ms_server"),
            f.get("reaction_ms_client"),
            f.get("pause_ratio"),
            f.get("speaking_rate_wps"),
            f.get("filler_count"),
            f.get("pitch_mean_hz"),
            r.created_at.isoformat(),
        ])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"},
    )

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","time":datetime.utcnow().isoformat()})

_PERF_WINDOW = []  # list of last N processing_ms
@app.after_request
def _collect_perf(resp):
    try:
        if request.path == '/api/analyze' and resp.is_json:
            data = resp.get_json(silent=True) or {}
            pm = data.get('processing_ms')
            if isinstance(pm, int):
                _PERF_WINDOW.append(pm)
                if len(_PERF_WINDOW) > 50:
                    _PERF_WINDOW.pop(0)
    except Exception:
        pass
    return resp

@app.route('/api/perf', methods=['GET'])
def perf():
    if _PERF_WINDOW:
        avg = sum(_PERF_WINDOW)/len(_PERF_WINDOW)
        p95 = sorted(_PERF_WINDOW)[int(0.95*len(_PERF_WINDOW))-1]
    else:
        avg = p95 = 0
    return jsonify({
        'recent_count': len(_PERF_WINDOW),
        'avg_ms': round(avg,1),
        'p95_ms': p95,
        'fast_mode': app.config.get('FAST_MODE', False)
    })

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # Use socketio.run to enable websockets
    socketio.run(app, debug=True, port=5000)
