// Fallos seguidos por "familia" de práctica (palabra completa, o sub-paso
// de RR como "perro_doble") -- usado para la dificultad que se ajusta sola
// (ver LearningPathScreen.jsx): si el chico falla varias veces seguidas en
// lo mismo, en vez de insistir con la evaluación real le ofrecemos una
// pausa de repetición libre (sin presión) antes de dejarlo reintentar.
// Guardado en localStorage -- no hace falta backend para esto, es solo
// ritmo de práctica de ESTE dispositivo.
const PREFIX = "fonokids_fallos_";

export const UMBRAL_DESCANSO = 3;

export function getFallosSeguidos(familia) {
  try {
    return parseInt(localStorage.getItem(PREFIX + familia) || "0", 10) || 0;
  } catch {
    return 0;
  }
}

/// Registra un intento -- si salió bien, resetea el contador a 0; si salió
/// mal, lo suma. Devuelve el nuevo total de fallos seguidos.
export function registrarIntento(familia, correcta) {
  const nuevo = correcta ? 0 : getFallosSeguidos(familia) + 1;
  try {
    localStorage.setItem(PREFIX + familia, String(nuevo));
  } catch {
    // localStorage puede fallar (modo privado, cuota) -- no bloquea la
    // práctica, solo no se guarda el contador para la próxima.
  }
  return nuevo;
}

export function resetFallos(familia) {
  try {
    localStorage.removeItem(PREFIX + familia);
  } catch {
    // ver comentario de arriba
  }
}
