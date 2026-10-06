// Texto a voz del navegador (Web Speech API) -- lo usan Aprender y el
// diagnóstico, centralizado acá para no duplicar la lógica de voces.
import { referenceAudioUrl } from "./api.js";

// El sistema no trae ninguna voz "de nene": las únicas voces en español
// disponibles son adultas (Helena/Laura/Pablo, Windows). Lo más cercano a
// algo más liviano/amigable sin sonar robótico es subir el "pitch" y el
// rate un toque, en vez de dejarlo plano y grave.
let cachedVoices = [];
if (typeof window !== "undefined" && window.speechSynthesis) {
  const loadVoices = () => {
    cachedVoices = window.speechSynthesis.getVoices();
  };
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}

/// Pedido explícito: si hay un clip de audio REAL de alguien diciendo la
/// palabra bien (ver referenceAudioUrl en api.js), reproducir ESE en vez de
/// una voz sintética -- más natural para el chico. La mayoría de las
/// palabras todavía no tienen un clip guardado, así que si el audio tira
/// error (404 u otro problema) cae directo al TTS de siempre, sin que se
/// note para quien usa la app.
export function playWordReference(baseUrl, palabra, textoParaTTS) {
  const audio = new Audio(referenceAudioUrl(baseUrl, palabra));
  let yaCayoATts = false;
  const caerATts = () => {
    if (yaCayoATts) return;
    yaCayoATts = true;
    speakWord(textoParaTTS);
  };
  audio.addEventListener("error", caerATts);
  audio.play().catch(caerATts);
}

export function speakWord(text) {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.cancel(); // corta cualquier lectura anterior sin terminar
  const utterance = new SpeechSynthesisUtterance(text);
  const spanishVoice = cachedVoices.find((v) => v.lang?.toLowerCase().startsWith("es"));
  if (spanishVoice) utterance.voice = spanishVoice;
  utterance.lang = spanishVoice?.lang || "es-ES";
  utterance.rate = 0.95;
  utterance.pitch = 1.35;
  window.speechSynthesis.speak(utterance);
}
