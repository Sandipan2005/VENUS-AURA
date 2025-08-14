class WavRecorder {
  constructor({ onData, onVADOnset, onSpeechEnd } = {}) {
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.proc = null;
    this.buffers = [];
    this.onData = onData;
    this.onVADOnset = onVADOnset;
    this.onSpeechEnd = onSpeechEnd;
    this.armed = false;
    this.onsetDetected = false;

    this.sampleRate = 48000;
    // simple VAD
    this.frameSize = 1024; // ~21 ms at 48k
    this.noiseFloor = 0.0;
    this.vadThreshold = 3.5; // lowered threshold for quicker onset
    this.consecNeeded = 4; // ~84 ms to detect speech
    this.consecCount = 0;
    // speech end detection
    this.afterOnsetSilenceFrames = 0;
    this.framesSinceOnset = 0;
    this.silenceNeededFramesEnd = 8; // ~170 ms silence to end
    this.minSpeechFrames = 8; // ~170 ms minimum speech before allowing end
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
      video: false,
    });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this.ctx.sampleRate;
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(this.frameSize, 1, 1);
    this.source.connect(this.proc);
    this.proc.connect(this.ctx.destination); // keep alive
    // calibrate noise floor for 150 ms (faster start)
    let calibrationFrames = Math.ceil(
      (0.15 * this.sampleRate) / this.frameSize
    );
    this.calFramesLeft = calibrationFrames;

    this.proc.onaudioprocess = (e) => {
      const buf = e.inputBuffer.getChannelData(0);
      // copy
      this.buffers.push(new Float32Array(buf));
      if (this.onData) this.onData(buf);

      // RMS
      let rms = 0;
      for (let i = 0; i < buf.length; i++) {
        rms += buf[i] * buf[i];
      }
      rms = Math.sqrt(rms / buf.length);

      if (this.calFramesLeft > 0) {
        this.noiseFloor =
          (this.noiseFloor * (calibrationFrames - this.calFramesLeft) + rms) /
          (calibrationFrames - this.calFramesLeft + 1e-9);
        this.calFramesLeft--;
        return;
      }

      if (this.armed) {
        const speechCond =
          rms > Math.max(this.noiseFloor * this.vadThreshold, 0.005);
        if (!this.onsetDetected) {
          if (speechCond) {
            this.consecCount++;
            if (this.consecCount >= this.consecNeeded) {
              this.onsetDetected = true;
              const t = this.ctx.currentTime * 1000.0; // ms
              if (this.onVADOnset) this.onVADOnset(t);
              this.framesSinceOnset = 0;
              this.afterOnsetSilenceFrames = 0;
            }
          } else {
            this.consecCount = 0;
          }
        } else {
          // already speaking
          this.framesSinceOnset++;
          if (speechCond) {
            this.afterOnsetSilenceFrames = 0; // reset silence
          } else {
            this.afterOnsetSilenceFrames++;
            if (
              this.afterOnsetSilenceFrames >= this.silenceNeededFramesEnd &&
              this.framesSinceOnset >= this.minSpeechFrames
            ) {
              // speech end
              this.armed = false; // prevent multiple triggers
              const tend = this.ctx.currentTime * 1000.0;
              if (this.onSpeechEnd) this.onSpeechEnd(tend);
            }
          }
        }
        // streaming: emit raw chunk (PCM16) for live metrics / partial ASR
        try {
          if (window._tldSocketEmit && this.armed) {
            // convert float32 buffer to PCM16 little endian
            const pcm16 = new Int16Array(buf.length);
            for (let i = 0; i < buf.length; i++) {
              let s = Math.max(-1, Math.min(1, buf[i]));
              pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            const b = new Uint8Array(pcm16.buffer);
            // base64 encode
            let bin = "";
            for (let i = 0; i < b.length; i++) bin += String.fromCharCode(b[i]);
            const b64 = btoa(bin);
            window._tldSocketEmit("audio_chunk", {
              client_id: window.localStorage.getItem("tld_client_id"),
              pcm16: b64,
              sample_rate: this.sampleRate,
              prompt_idx: window._tldPromptIndex || 0,
            });
          }
        } catch (e) {}
      }
    };
  }

  arm() {
    this.armed = true;
    this.onsetDetected = false;
    this.consecCount = 0;
  }

  stop() {
    if (this.proc) this.proc.disconnect();
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();
  }

  exportWav() {
    // flatten
    const length = this.buffers.reduce((a, b) => a + b.length, 0);
    const data = new Float32Array(length);
    let offset = 0;
    this.buffers.forEach((b) => {
      data.set(b, offset);
      offset += b.length;
    });
    // PCM16
    const buffer = new ArrayBuffer(44 + data.length * 2);
    const view = new DataView(buffer);

    const writeString = (off, s) => {
      for (let i = 0; i < s.length; i++)
        view.setUint8(off + i, s.charCodeAt(i));
    };
    const write16 = (off, v) => view.setUint16(off, v, true);
    const write32 = (off, v) => view.setUint32(off, v, true);

    // RIFF header
    writeString(0, "RIFF");
    write32(4, 36 + data.length * 2);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    write32(16, 16); // PCM
    write16(20, 1); // linear PCM
    write16(22, 1); // channels
    write32(24, this.sampleRate);
    write32(28, this.sampleRate * 2);
    write16(32, 2);
    write16(34, 16);
    writeString(36, "data");
    write32(40, data.length * 2);

    // samples
    let idx = 44;
    for (let i = 0; i < data.length; i++, idx += 2) {
      let s = Math.max(-1, Math.min(1, data[i]));
      view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: "audio/wav" });
  }
}
