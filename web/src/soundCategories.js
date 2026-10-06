// Categorías de sonidos difíciles del español (dislalias más comunes en
// chicos) -- fuente única compartida entre SoundsScreen (repetición libre,
// texto a voz) y LearningPathScreen (Aprender: categorías -> etapas ->
// pasos), para que las dos pantallas siempre muestren las mismas
// categorías y no se desincronicen entre sí.
//
// Pedido explícito: solo palabras con clases REALMENTE entrenadas (ver
// data/audio_embeddings_cache/ y data/landmarks_npy/) -- nada de "zorro",
// "jarra", etc. que suenan bien pero no tienen ni un clip grabado. Por eso
// no están acá las categorías R suave, CH/LL-Y, J/G fuerte, F ni
// Consonantes al final: ninguna palabra de esas tiene una sola clase
// entrenada todavía, quedan afuera hasta que se graben de verdad (en vez
// de mostrarlas como relleno sin datos).
//
// Orden de MENOS a MÁS difícil (según cuándo se adquiere normalmente cada
// sonido en español y qué tan común/persistente es la dislalia): C/K y G
// suave (velares, se resuelven solas casi siempre entre los 2-4 años) ->
// Sinfones (grupos consonánticos, reducción común en preescolares) ->
// S (sigmatismo, persiste más) -> RR fuerte (la más común y la que más
// tarda en adquirirse, normal recién a los 4-5 años).
export const SOUND_CATEGORIES = [
  {
    key: "c-k",
    title: "C / K",
    hint: "sonido de atrás de la boca, sin vibración -- a veces se cambia por T",
    words: [
      { word: "casa", icon: "🏠", sound: "k" },
      { word: "cama", icon: "🛏️", sound: "k" },
      { word: "cohete", icon: "🚀", sound: "k" },
      { word: "copa", icon: "🏆", sound: "k" },
      { word: "cubo", icon: "🧊", sound: "k" },
    ],
  },
  {
    key: "g",
    title: "G suave",
    hint: "sonido de atrás de la boca, con vibración -- a veces se cambia por D",
    words: [
      { word: "gato", icon: "🐱", sound: "g" },
      { word: "goma", icon: "🧽", sound: "g" },
      { word: "gota", icon: "💧", sound: "g" },
      { word: "gusano", icon: "🐛", sound: "g" },
    ],
  },
  {
    key: "sinfones",
    title: "Sinfones (dos consonantes seguidas)",
    hint: "la boca cambia rápido de posición sin meter una vocal en el medio",
    words: [
      { word: "blanco", icon: "⬜", sound: "bl" },
      { word: "flor", icon: "🌸", sound: "fl" },
      { word: "globo", icon: "🎈", sound: "gl" },
      { word: "platano", label: "plátano", icon: "🍌", sound: "pl" },
      { word: "clavo", icon: "🔨", sound: "cl" },
    ],
  },
  {
    key: "s",
    title: "S (sigmatismo)",
    hint: "la lengua se va entre los dientes y suena como una Z",
    words: [
      { word: "sandia", label: "sandía", icon: "🍉", sound: "s" },
      { word: "sapo", icon: "🐸", sound: "s" },
      { word: "sopa", icon: "🍲", sound: "s" },
      { word: "serpiente", icon: "🐍", sound: "s" },
      { word: "silla", icon: "🪑", sound: "s" },
    ],
  },
  {
    key: "rr",
    title: "RR fuerte",
    hint: "lengua vibrando, sonido bien marcado",
    words: [
      { word: "perro", icon: "🐶", sound: "rr" },
      { word: "carro", icon: "🚗", sound: "rr" },
      { word: "torre", icon: "🗼", sound: "rr" },
      { word: "burro", icon: "🫏", sound: "rr" },
      { word: "gorra", icon: "🧢", sound: "rr" },
    ],
  },
  {
    key: "ll-ch",
    title: "LL / CH",
    hint: "la lengua toca el paladar -- a veces se cambia por Y suave o se pierde el golpe de la CH",
    words: [
      { word: "llave", icon: "🔑", sound: "ll" },
      { word: "lluvia", icon: "🌧️", sound: "ll" },
      { word: "pollo", icon: "🐔", sound: "ll" },
      { word: "coche", icon: "🚙", sound: "ch" },
    ],
  },
];
