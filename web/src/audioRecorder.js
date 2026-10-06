// Grabador de audio -> WAV real (16 bits, mono) usando Web Audio API --
// extraído de HomeScreen.jsx para reusarlo también en FrasesScreen.jsx sin
// duplicar la lógica. El servidor (server/main.py) espera WAV de verdad,
// no el webm/opus que da MediaRecorder por defecto -- por eso se arma el
// header RIFF/WAVE a mano en vez de usar MediaRecorder para el audio.

function encodeWav(chunks, sampleRate) {
  let totalLength = 0;
  for (const c of chunks) totalLength += c.length;

  const pcm16 = new Int16Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const s = Math.max(-1, Math.min(1, chunk[i]));
      pcm16[offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }

  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);
  const writeStr = (pos, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(pos + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + pcm16.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, pcm16.length * 2, true);
  new Int16Array(buffer, 44).set(pcm16);

  return new Blob([buffer], { type: "audio/wav" });
}

/// Crea un grabador independiente (sin estado de React) -- start(stream)
/// arranca a juntar PCM crudo del track de audio del stream, stop()
/// corta y devuelve el Blob .wav ya armado (o null si no había audio).
export function createAudioRecorder() {
  let ctx = null;
  let source = null;
  let processor = null;
  let chunks = [];

  function start(stream) {
    const audioTrack = stream?.getAudioTracks?.()[0];
    if (!audioTrack) return false;
    chunks = [];
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      ctx = new AudioCtx();
      source = ctx.createMediaStreamSource(new MediaStream([audioTrack]));
      processor = ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      return true;
    } catch (e) {
      console.warn("No se pudo iniciar la captura de audio:", e);
      return false;
    }
  }

  function stop() {
    if (!ctx) return null;
    const sampleRate = ctx.sampleRate;
    try {
      processor?.disconnect();
      source?.disconnect();
      ctx.close();
    } catch {
      // no crítico
    }
    const finalChunks = chunks;
    ctx = null;
    source = null;
    processor = null;
    chunks = [];
    if (finalChunks.length === 0) return null;
    return encodeWav(finalChunks, sampleRate);
  }

  return { start, stop };
}
