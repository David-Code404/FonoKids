// Categorías de sonidos difíciles del español (dislalias más comunes en
// chicos) -- fuente única compartida entre SoundsScreen (repetición libre,
// texto a voz) y LearningPathScreen (Aprender: categorías -> etapas ->
// pasos), para que las dos pantallas siempre muestren las mismas
// categorías y no se desincronicen entre sí.
//
// Solo las palabras de RR fuerte tienen clases entrenadas hoy (correcto /
// incorrecto con omisión, lambdacismo, dentalización) -- las demás quedan
// como práctica libre con voz (texto a voz) hasta que se graben y
// entrenen sus propias clases.
export const SOUND_CATEGORIES = [
  {
    key: "r",
    title: "R suave",
    hint: "un solo golpecito de lengua, sin vibrar",
    words: [
      { word: "raton", icon: "🐭" },
      { word: "mariposa", icon: "🦋" },
      { word: "tren", icon: "🚂" },
      { word: "fresa", icon: "🍓" },
      { word: "rueda", icon: "🛞" },
      { word: "sombrero", icon: "🎩" },
      { word: "bicicleta", icon: "🚲" },
      { word: "pera", icon: "🍐" },
      { word: "cara", icon: "🙂" },
    ],
  },
  {
    key: "rr",
    title: "RR fuerte",
    hint: "lengua vibrando, sonido bien marcado",
    words: [
      { word: "perro", icon: "🐶" },
      { word: "carro", icon: "🚗" },
      { word: "torre", icon: "🗼" },
      { word: "burro", icon: "🫏" },
      { word: "gorra", icon: "🧢" },
      { word: "jarra", icon: "🏺" },
      { word: "barra", icon: "📏" },
      { word: "guerra", icon: "⚔️" },
      { word: "arroz", icon: "🍚" },
      { word: "zorro", icon: "🦊" },
    ],
  },
  {
    key: "s",
    title: "S (sigmatismo)",
    hint: "la lengua se va entre los dientes y suena como una Z",
    words: [
      { word: "sol", icon: "☀️" },
      { word: "sapo", icon: "🐸" },
      { word: "sirena", icon: "🧜‍♀️" },
      { word: "sandia", icon: "🍉" },
      { word: "salchicha", icon: "🌭" },
      { word: "sopa", icon: "🍲" },
      { word: "sombra", icon: "🌑" },
      { word: "silbato", icon: "📯" },
      { word: "seis", icon: "6️⃣" },
    ],
  },
  {
    key: "sinfones",
    title: "Sinfones (dos consonantes seguidas)",
    hint: "la boca cambia rápido de posición sin meter una vocal en el medio",
    words: [
      { word: "blanco", icon: "⬜" },
      { word: "flor", icon: "🌸" },
      { word: "tren", icon: "🚂" },
      { word: "fresa", icon: "🍓" },
      { word: "premio", icon: "🎁" },
      { word: "clase", icon: "🏫" },
      { word: "globo", icon: "🎈" },
      { word: "platano", icon: "🍌" },
      { word: "cruz", icon: "✝️" },
      { word: "grillo", icon: "🦗" },
    ],
  },
  {
    key: "ch-ll",
    title: "CH / LL-Y",
    hint: "necesitan que la lengua presione fuerte contra el paladar",
    words: [
      { word: "chocolate", icon: "🍫" },
      { word: "muchacho", icon: "🧒" },
      { word: "lluvia", icon: "🌧️" },
      { word: "llave", icon: "🔑" },
      { word: "pollo", icon: "🐓" },
      { word: "chancho", icon: "🐷" },
      { word: "silla", icon: "🪑" },
      { word: "estrella", icon: "⭐" },
    ],
  },
  {
    key: "l",
    title: "L",
    hint: "la lengua sube y toca detrás de los dientes de arriba",
    words: [
      { word: "luna", icon: "🌙" },
      { word: "leon", icon: "🦁" },
      { word: "lapiz", icon: "✏️" },
      { word: "limon", icon: "🍋" },
      { word: "libro", icon: "📖" },
      { word: "lobo", icon: "🐺" },
      { word: "loro", icon: "🦜" },
      { word: "luz", icon: "💡" },
      { word: "lazo", icon: "🎀" },
    ],
  },
  {
    key: "g",
    title: "G suave",
    hint: "sonido de atrás de la boca, con vibración -- a veces se cambia por D",
    words: [
      { word: "gato", icon: "🐱" },
      { word: "guitarra", icon: "🎸" },
      { word: "gusano", icon: "🐛" },
      { word: "gallina", icon: "🐔" },
      { word: "gorila", icon: "🦍" },
      { word: "gafas", icon: "👓" },
      { word: "goma", icon: "🧽" },
      { word: "gol", icon: "⚽" },
      { word: "golosina", icon: "🍬" },
    ],
  },
  {
    key: "c-k",
    title: "C / K",
    hint: "sonido de atrás de la boca, sin vibración -- a veces se cambia por T",
    words: [
      { word: "casa", icon: "🏠" },
      { word: "cohete", icon: "🚀" },
      { word: "koala", icon: "🐨" },
      { word: "castillo", icon: "🏰" },
      { word: "cocodrilo", icon: "🐊" },
      { word: "cama", icon: "🛏️" },
      { word: "conejo", icon: "🐰" },
      { word: "copa", icon: "🏆" },
      { word: "cabra", icon: "🐐" },
    ],
  },
  {
    key: "j-g-fuerte",
    title: "J / G fuerte",
    hint: "se raspa el aire en la parte de atrás de la garganta",
    words: [
      { word: "jirafa", icon: "🦒" },
      { word: "ojo", icon: "👁️" },
    ],
  },
  {
    key: "f",
    title: "F",
    hint: "el labio de abajo tiene que apoyarse en los dientes de arriba",
    words: [
      { word: "foco", icon: "💡" },
    ],
  },
  {
    key: "finales",
    title: "Consonantes al final",
    hint: "las últimas letras de la palabra se comen o cambian",
    words: [
      { word: "reloj", icon: "⏰" },
      { word: "actor", icon: "🎭" },
    ],
  },
];
