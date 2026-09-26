// Texto a voz del navegador (Web Speech API) -- compartido entre
// SoundsScreen y LearningPathScreen para no duplicar la lógica de voces.
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
