// Una frase natural por cada palabra entrenada -- para el modo "practicar
// con frases" (ver FrasesScreen.jsx). La palabra objetivo aparece siempre
// tal cual, en algún lugar de la frase (server/main.py la ubica sola
// dentro de la frase completa, ver score_word_in_sentence).
export const FRASES_POR_PALABRA = {
  perro: "El perro corre por el parque",
  carro: "El carro es de color rojo",
  torre: "La torre es muy alta",
  burro: "El burro come pasto",
  gorra: "Me puse la gorra azul",
  sandia: "La sandía es dulce y roja",
  sapo: "El sapo salta en el charco",
  sopa: "Me gusta comer sopa caliente",
  serpiente: "La serpiente se mueve despacio",
  silla: "Me siento en la silla",
  blanco: "El gato es blanco",
  flor: "La flor huele muy bien",
  globo: "El globo vuela alto",
  platano: "El plátano es amarillo",
  clavo: "El clavo está en la pared",
  gato: "El gato duerme en el sofá",
  goma: "Borré con la goma",
  gota: "Cayó una gota de agua",
  gusano: "El gusano se arrastra despacio",
  casa: "Mi casa es grande",
  cama: "Duermo en mi cama",
  cohete: "El cohete vuela al espacio",
  copa: "Bebí jugo en la copa",
  cubo: "El cubo tiene seis caras",
};

export function fraseParaPalabra(palabra) {
  return FRASES_POR_PALABRA[palabra] || palabra;
}

// Algunas palabras objetivo no llevan tilde ("sandia", "platano") pero la
// frase SÍ la escribe bien acentuada ("sandía", "plátano") -- un regex
// literal de la palabra no la encontraba dentro de la frase (día != dia),
// así que el resaltado se rompía. Esto arma un regex que ignora tilde en
// las vocales, para encontrar/resaltar la palabra tenga o no tenga acento.
const _VOCAL_CON_TILDE = { a: "aá", e: "eé", i: "ií", o: "oó", u: "uúü" };
export function regexPalabraEnFrase(palabra) {
  const patron = palabra
    .split("")
    .map((ch) => {
      const clases = _VOCAL_CON_TILDE[ch.toLowerCase()];
      return clases ? `[${clases}]` : ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    })
    .join("");
  return new RegExp(`(${patron})`, "i");
}
