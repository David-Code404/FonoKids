// Progreso guardado en el navegador (sirve aunque MySQL esté apagado): ejercicios
// hechos, días de uso, niveles logrados y evaluaciones previa / intermedia / final.

const K_HECHOS = "fk_ejercicios_hechos";
const K_DIAS = "fk_dias_de_uso";
const K_LOGRADAS = "fk_palabras_logradas";
const K_EVALS = "fk_evaluaciones";
const K_SOLO_IMAGEN = "fk_solo_imagen";

function leer(clave, porDefecto) {
  try {
    const v = localStorage.getItem(clave);
    return v ? JSON.parse(v) : porDefecto;
  } catch {
    return porDefecto;
  }
}
function escribir(clave, valor) {
  try {
    localStorage.setItem(clave, JSON.stringify(valor));
  } catch { /* sin almacenamiento: la app sigue funcionando */ }
}

export function registrarEjercicioHecho() {
  escribir(K_HECHOS, leer(K_HECHOS, 0) + 1);
  const hoy = new Date().toISOString().slice(0, 10);
  const dias = leer(K_DIAS, []);
  if (!dias.includes(hoy)) escribir(K_DIAS, [...dias, hoy]);
}
export const ejerciciosHechos = () => leer(K_HECHOS, 0);
export const diasDeUso = () => leer(K_DIAS, []).length;

export function marcarLograda(palabra) {
  const l = leer(K_LOGRADAS, {});
  l[palabra] = true;
  escribir(K_LOGRADAS, l);
}
export const palabrasLogradas = () => leer(K_LOGRADAS, {});

export const soloImagen = () => leer(K_SOLO_IMAGEN, false);
export const guardarSoloImagen = (v) => escribir(K_SOLO_IMAGEN, !!v);

// evaluaciones: { r: { previa: {bien, total, t, hechos}, intermedia: {...}, final: {...} } }
export const evaluaciones = () => leer(K_EVALS, {});
export function guardarEvaluacion(sonido, etapa, bien, total, logradas = 0) {
  const e = evaluaciones();
  e[sonido] = { ...(e[sonido] || {}), [etapa]: { bien, total, t: Date.now(), hechos: ejerciciosHechos(), logradas } };
  escribir(K_EVALS, e);
}

/// Empezar de nuevo un sonido: se borran las palabras logradas y las pruebas intermedia y final
/// (se conserva la previa, que es el punto de partida del niño).
export function reiniciarSonido(sonido, palabras) {
  const l = leer(K_LOGRADAS, {});
  palabras.forEach((p) => { delete l[p]; });
  escribir(K_LOGRADAS, l);
  const e = evaluaciones();
  if (e[sonido]) {
    const { previa } = e[sonido];
    e[sonido] = previa ? { previa } : {};
    escribir(K_EVALS, e);
  }
}

/// Qué evaluación toca para un sonido: "previa" si nunca se hizo; "intermedia"
/// cuando pasaron `cada` ejercicios desde la previa; "final" cuando pasaron
/// `cada` más desde la intermedia; null si ya terminó o todavía no toca.
export function evaluacionQueToca(sonido, cada) {
  const e = evaluaciones()[sonido] || {};
  const hechos = ejerciciosHechos();
  if (!e.previa) return "previa";
  if (!e.intermedia) return hechos - e.previa.hechos >= cada ? "intermedia" : null;
  if (!e.final) return hechos - e.intermedia.hechos >= cada ? "final" : null;
  return null;
}
export function ejerciciosParaSiguiente(sonido, cada) {
  const e = evaluaciones()[sonido] || {};
  const ref = e.intermedia || e.previa;
  if (!ref || e.final) return 0;
  return Math.max(cada - (ejerciciosHechos() - ref.hechos), 0);
}

const K_CAMARA = "fk_usar_camara";
export const usarCamara = () => leer(K_CAMARA, true);
export const guardarUsarCamara = (v) => escribir(K_CAMARA, !!v);

// ---- Evaluaciones por PALABRAS logradas (como pidió la fonoaudióloga) ----
/// Cuántas de las palabras de la lista ya se lograron.
export const contarLogradas = (lista) => {
  const l = palabrasLogradas();
  return lista.filter((p) => l[p]).length;
};

/// Estado de las tres evaluaciones de un sonido:
///  - previa: siempre disponible hasta hacerla;
///  - intermedia: se abre al lograr `cada` palabras más desde la previa;
///  - final: se abre al lograr TODAS las palabras del sonido (y haber hecho la intermedia).
// Sonidos con TODO habilitado para la presentación (pruebas y palabras abiertas sin condiciones).
export const SONIDOS_DESBLOQUEADOS = new Set(["m"]);

export function estadoEvaluacion(sonido, palabras, cada = 5) {
  const e = evaluaciones()[sonido] || {};
  if (SONIDOS_DESBLOQUEADOS.has(sonido)) {
    const logradas = contarLogradas(palabras);
    return {
      e: { ...e, previa: e.previa || true, intermedia: e.intermedia || true }, logradas, total: palabras.length,
      previa: { hecha: e.previa || null, toca: true },
      intermedia: { hecha: e.intermedia || null, toca: true, faltan: 0 },
      final: { hecha: e.final || null, toca: true, faltan: 0 },
    };
  }
  // Una prueba del medio o final con menos de 2 estrellas no cuenta: se empieza de nuevo el sonido
  // (también limpia lo que se guardó antes de existir esta regla).
  if ((e.intermedia && e.intermedia.bien < 2) || (e.final && e.final.bien < 2)) {
    reiniciarSonido(sonido, palabras);
    return estadoEvaluacion(sonido, palabras, cada);
  }
  const logradas = contarLogradas(palabras);
  const total = palabras.length;
  const desdePrevia = logradas - (e.previa?.logradas ?? 0);
  return {
    e, logradas, total,
    previa: { hecha: e.previa || null, toca: !e.previa },
    intermedia: { hecha: e.intermedia || null, toca: !e.intermedia && desdePrevia >= cada, faltan: Math.max(cada - desdePrevia, 0) },
    final: { hecha: e.final || null, toca: !e.final && logradas >= total, faltan: Math.max(total - logradas, 0) },
  };
}
