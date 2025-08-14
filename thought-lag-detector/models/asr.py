
from faster_whisper import WhisperModel
import os, time

class LazyWhisper:
    def __init__(self, model_name: str, device: str = "cpu", compute_type: str = "int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self.loaded_at = None

    @property
    def model(self):
        if self._model is None:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            self.loaded_at = time.time()
        return self._model

class ASR:
	def __init__(self, model_name: str = "tiny.en"):
		# Primary low-latency model
		self.primary = WhisperModel(model_name, device="cpu", compute_type="int8")
		self.refine_lazy: LazyWhisper | None = None
		self.prev_words: list[str] = []  # for partial diff
		self.partial_last_text = ""

	def transcribe(self, wav_path: str):
		segments, info = self.primary.transcribe(
			wav_path,
			language="en",
			vad_filter=True,
			vad_parameters={"min_silence_duration_ms": 200},
			beam_size=3,
		)
		return self._segments_to_result(segments)

	def partial_transcribe(self, wav_path: str, max_tokens: int = 64):
		"""Fast partial decode (no condition on previous text, beam_size=1)."""
		segments, info = self.primary.transcribe(
			wav_path,
			language="en",
			vad_filter=False,
			beam_size=1,
			condition_on_previous_text=False,
		)
		res = self._segments_to_result(segments)
		# Diff new words
		new_words = []
		for w in res["words"]:
			if w["word"] not in self.prev_words:
				new_words.append(w["word"])
		# Update history limited size
		self.prev_words.extend(new_words)
		if len(self.prev_words) > 400:
			self.prev_words = self.prev_words[-400:]
		return {"text": res["text"], "words": res["words"], "new_text": " ".join(new_words)}

	def refine_transcribe(self, wav_path: str, model_name: str, device: str = "cpu"):
		if not self.refine_lazy or self.refine_lazy.model_name != model_name:
			self.refine_lazy = LazyWhisper(model_name, device=device, compute_type="int8")
		segments, info = self.refine_lazy.model.transcribe(
			wav_path,
			language="en",
			vad_filter=True,
			beam_size=5,
		)
		return self._segments_to_result(segments)

	def _segments_to_result(self, segments):
		text = []
		words = []
		for seg in segments:
			text.append(seg.text)
			if seg.words:
				for w in seg.words:
					words.append({"word": w.word, "start": w.start, "end": w.end})
		return {"text": " ".join(text).strip(), "words": words}
