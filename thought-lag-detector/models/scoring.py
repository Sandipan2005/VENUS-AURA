
import numpy as np
from collections import defaultdict
try:
	from .storage import db, BaselineStat
except Exception:  # fallback if import cycle
	BaselineStat = None  # type: ignore
_BASELINES = {}
_HISTORY = defaultdict(list)

FEATURE_KEYS = [
	"reaction_ms_server",
	"pause_ratio",
	"pitch_cv",
	"intensity_var",
	"speaking_rate_wps",
	"filler_count",
	"jitter_local",
	"shimmer_local",
	"breath_hf_ratio",
]

class ScoreModel:
	def __init__(self):
		pass
	def _update_baseline(self, client_id, feats):
		hist = _HISTORY[client_id]
		hist.append(feats)
		if len(hist) < 6:
			return
		means = {}
		stds = {}
		for k in FEATURE_KEYS:
			vals = [h.get(k, 0.0) for h in hist]
			if not vals:
				continue
			means[k] = float(np.mean(vals))
			std = float(np.std(vals)) or 1.0
			stds[k] = std
		_BASELINES[client_id] = {"means": means, "stds": stds, "count": len(hist)}
		# persist
		if BaselineStat is not None:
			try:
				row = BaselineStat.query.filter_by(client_id=client_id).first()
				if not row:
					row = BaselineStat()
					row.client_id = client_id
				row.feature_means = means
				row.feature_stds = stds
				row.count = len(hist)
				db.session.add(row)  # type: ignore
				db.session.commit()  # type: ignore
			except Exception:
				pass

	def score(self, client_id, feats):
		# Update baseline incrementally (or build a UI flow for "baseline" session)
		self._update_baseline(client_id, feats)
		base = _BASELINES.get(client_id)
		if not base and BaselineStat is not None:
			try:
				row = BaselineStat.query.filter_by(client_id=client_id).first()
				if row:
					base = {"means": row.feature_means or {}, "stds": row.feature_stds or {}, "count": row.count}
					_BASELINES[client_id] = base
			except Exception:
				pass
		# fallbacks if baseline not ready
		if not base:
			base = {
				"means": {
					"reaction_ms_server": 600.0,
					"pause_ratio": 0.25,
					"pitch_cv": 0.12,
					"intensity_var": 1e-3,
					"speaking_rate_wps": 2.0,
					"filler_count": 1.0,
					"jitter_local": 0.02,
					"shimmer_local": 0.03,
					"breath_hf_ratio": 0.12,
				},
				"stds": {
					"reaction_ms_server": 150.0,
					"pause_ratio": 0.1,
					"pitch_cv": 0.05,
					"intensity_var": 1e-3,
					"speaking_rate_wps": 0.6,
					"filler_count": 1.0,
					"jitter_local": 0.01,
					"shimmer_local": 0.015,
					"breath_hf_ratio": 0.05,
				}
			}
		z = {}

		def zscore(k, invert=False):
			m = base["means"].get(k, 0.0)
			s = base["stds"].get(k, 1.0)
			val = feats.get(k, 0.0)
			zval = (val - m) / (s if s != 0 else 1.0)
			return -zval if invert else zval

		# Higher stress when: higher reaction, higher pause_ratio, higher pitch_cv,
		# higher intensity_var (variability), more fillers, speaking_rate lower than baseline (invert)
		# plus higher jitter/shimmer and higher breath ratio (proxy tension)
		z["z_reaction"] = zscore("reaction_ms_server")
		z["z_pause"] = zscore("pause_ratio")
		z["z_pitchcv"] = zscore("pitch_cv")
		z["z_intvar"] = zscore("intensity_var")
		z["z_fillers"] = zscore("filler_count")
		z["z_rate"] = zscore("speaking_rate_wps", invert=True)
		z["z_jitter"] = zscore("jitter_local")
		z["z_shimmer"] = zscore("shimmer_local")
		z["z_breath"] = zscore("breath_hf_ratio")

		# Weighted sum -> logistic -> 0-100
		w = {
			"z_reaction": 0.28,
			"z_pause": 0.16,
			"z_pitchcv": 0.14,
			"z_intvar": 0.08,
			"z_fillers": 0.08,
			"z_rate": 0.12,
			"z_jitter": 0.07,
			"z_shimmer": 0.04,
			"z_breath": 0.03,
		}
		# contribution map
		contrib = {k: w[k] * z.get(k, 0.0) for k in w}
		s = sum(contrib.values())
		stress = float(100.0 / (1.0 + np.exp(-s)))  # 0..100
		focus = float(100.0 - stress + 8.0 * max(-z["z_fillers"], 0) + 6.0 * max(-z["z_pause"], 0) + 4.0 * max(-z["z_reaction"], 0))
		focus = float(np.clip(focus, 0, 100))
		# top drivers by absolute contribution
		drivers = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
		return {
			"stress": stress,
			"focus": focus,
			"z": z,
			"baseline_ready": bool(base),
			"contrib": contrib,
			"drivers": drivers,
			"baseline_means": base.get("means", {}),
			"baseline_stds": base.get("stds", {}),
		}
