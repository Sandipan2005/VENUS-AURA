
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from collections import deque

def read_wav_mono(path, target_sr=16000):
	y, sr = sf.read(path, dtype="float32", always_2d=False)
	if y.ndim > 1:
		y = np.mean(y, axis=1)
	if sr != target_sr:
		y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
		sr = target_sr
	# Clip to [-1,1]
	y = np.clip(y, -1.0, 1.0)
	return y, sr

def float_to_pcm16(y):
	y = np.clip(y, -1.0, 1.0)
	return (y * 32767.0).astype(np.int16).tobytes()

def webrtc_vad_segments(y, sr=16000, frame_ms=20, mode=2):
	vad = webrtcvad.Vad(mode)
	frame_len = int(sr * frame_ms / 1000)
	hop = frame_len # non-overlap
	pcm = float_to_pcm16(y)
	segments = []
	voiced_flags = []
	offs = 0
	for i in range(0, len(pcm), frame_len * 2): # 2 bytes per sample
		frame = pcm[i:i + frame_len * 2]
		if len(frame) < frame_len * 2:
			break
		is_speech = vad.is_speech(frame, sr)
		t0 = (i // 2) / sr
		t1 = t0 + frame_len / sr
		voiced_flags.append((t0, t1, is_speech))
	# merge flags to segments
	cur = None
	for t0, t1, sp in voiced_flags:
		if sp and cur is None:
			cur = [t0, t1]
		elif sp and cur is not None:
			cur[1] = t1
		elif (not sp) and cur is not None:
			segments.append(tuple(cur))
			cur = None
	if cur is not None:
		segments.append(tuple(cur))
	return segments, voiced_flags

def compute_pitch(y, sr):
	fmin = float(librosa.note_to_hz('C2'))
	fmax = float(librosa.note_to_hz('C7'))
	f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
	f0_hz = f0[~np.isnan(f0)] if f0 is not None else np.array([])
	if f0 is None or len(f0_hz) == 0:
		return {"pitch_mean_hz": 0.0, "pitch_cv": 0.0, "f0_series": f0}
	mean = float(np.mean(f0_hz))
	std = float(np.std(f0_hz))
	cv = float(std / (mean + 1e-6))
	return {"pitch_mean_hz": mean, "pitch_cv": cv, "f0_series": f0}

def compute_intensity_variance(y):
	# simple RMS variance
	frame = 2048
	hop = 512
	rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
	return float(np.var(rms))

def compute_pause_stats(voiced_flags):
	# overall duration and voiced/unvoiced totals within speech response
	if not voiced_flags:
		return {"pause_ratio": 1.0, "mean_pause_ms": 0.0, "speech_duration_sec": 0.0}
	t_min = voiced_flags[0][0]
	t_max = voiced_flags[-1][1]
	total = t_max - t_min
	pauses = []
	last_end = None
	for t0, t1, sp in voiced_flags:
		if last_end is None:
			last_end = t1 if sp else None
			continue
		if sp:
			if last_end is not None and t0 > last_end:
				pauses.append(t0 - last_end)
			last_end = t1
		else:
			pass
	if total <= 0:
		return {"pause_ratio": 1.0, "mean_pause_ms": 0.0, "speech_duration_sec": 0.0}
	pause_total = float(np.sum(pauses)) if pauses else 0.0
	pause_ratio = pause_total / total if total > 0 else 1.0
	mean_pause_ms = float(np.mean(pauses) * 1000.0) if pauses else 0.0
	# speech duration: sum of voiced segments length
	speech_dur = 0.0
	cur = None
	for t0, t1, sp in voiced_flags:
		if sp:
			speech_dur += (t1 - t0)
	return {"pause_ratio": float(pause_ratio), "mean_pause_ms": mean_pause_ms, "speech_duration_sec": float(speech_dur)}

def analyze_wav(path, vad_mode=2, fast_mode=False):
	y, sr = read_wav_mono(path, target_sr=16000)
	segments, flags = webrtc_vad_segments(y, sr=sr, frame_ms=20, mode=vad_mode)
	pause_stats = compute_pause_stats(flags)
	if fast_mode:
		pitch_mean = 0.0
		pitch_cv = 0.0
		intensity_var = 0.0
		jitter = 0.0
		shimmer = 0.0
		breath_ratio = 0.0
	else:
		pitch = compute_pitch(y, sr)
		pitch_mean = pitch["pitch_mean_hz"]
		pitch_cv = pitch["pitch_cv"]
		intensity_var = compute_intensity_variance(y)
		# Jitter / Shimmer (very approximate)
		f0_series = pitch.get("f0_series") if isinstance(pitch, dict) else None
		jitter = 0.0
		shimmer = 0.0
		if f0_series is not None and hasattr(f0_series, "__len__"):
			try:
				f0_valid = f0_series[~np.isnan(f0_series)]
				if len(f0_valid) > 5:
					periods = 1.0 / np.clip(f0_valid, 1e-3, None)
					jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-6))
			except Exception:
				pass
		# shimmer: use amplitude envelope via short RMS sequence
		try:
			frame = 1024
			hop = 512
			env = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
			if len(env) > 5:
				shimmer = float(np.mean(np.abs(np.diff(env))) / (np.mean(env) + 1e-6))
		except Exception:
			pass
		# Breath / high-frequency noise proxy: ratio of HF (>4k) energy to total during unvoiced zones near edges
		breath_ratio = 0.0
		try:
			S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
			freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
			hf_mask = freqs > 4000
			total_e = np.sum(S)
			hf_e = np.sum(S[hf_mask, :])
			breath_ratio = float(hf_e / (total_e + 1e-9))
		except Exception:
			pass
	duration_sec = len(y) / sr
	return {
		"vad_segments": segments,
		"duration_sec": float(duration_sec),
		"pitch_mean_hz": pitch_mean,
		"pitch_cv": pitch_cv,
		"intensity_var": float(intensity_var),
		"jitter_local": float(jitter),
		"shimmer_local": float(shimmer),
		"breath_hf_ratio": float(breath_ratio),
		"pause_ratio": pause_stats["pause_ratio"],
		"mean_pause_ms": pause_stats["mean_pause_ms"],
		"speech_duration_sec": pause_stats["speech_duration_sec"],
	}

def refine_reaction_with_vad(vad_segments, t_prompt_end_ms):
	# find the first VAD segment that starts AFTER prompt end
	t_end_s = t_prompt_end_ms / 1000.0
	candidates = [seg for seg in vad_segments if seg[0] >= t_end_s]
	if not candidates:
		return None
	onset = candidates[0][0] * 1000.0
	return float(onset - t_prompt_end_ms)
