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
export function guardarEvaluacion(sonido, etapa, bien, total) {
  const e = evaluaciones();
  e[sonido] = { ...(e[sonido] || {}), [etapa]: { bien, total, t: Date.now(), hechos: ejerciciosHechos() } };
  escribir(K_EVALS, e);
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
