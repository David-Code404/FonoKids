// Rachas y estrellas -- se guardan en localStorage (igual que el
// diagnóstico y los fallos seguidos), así que son de ESTE dispositivo.
//
// Reglas:
//   - Una estrella por cada intento VÁLIDO que salió bien dicho.
//   - La racha cuenta días seguidos con al menos un intento válido (bien o
//     mal): practicar es lo que mantiene la racha, no acertar -- a un chico
//     no se lo castiga por equivocarse.
//   - Si pasa un día entero sin practicar, la racha vuelve a cero.
const KEY = "fonokids_rachas";

function diaStr(d) {
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

function hoyStr() {
  return diaStr(new Date());
}

function ayerStr() {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return diaStr(d);
}

function leer() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { estrellas: 0, racha: 0, mejorRacha: 0, ultimoDia: null, ...JSON.parse(raw) };
  } catch {
    // localStorage bloqueado o JSON roto -- se arranca de cero.
  }
  return { estrellas: 0, racha: 0, mejorRacha: 0, ultimoDia: null };
}

function guardar(estado) {
  try {
    localStorage.setItem(KEY, JSON.stringify(estado));
  } catch {
    // sin persistencia, la app sigue andando igual.
  }
}

/// Estado para mostrar en pantalla. La racha guardada puede estar vencida
/// (último día de práctica anterior a ayer) -- en ese caso se muestra 0.
export function getEstado() {
  const e = leer();
  const vigente = e.ultimoDia === hoyStr() || e.ultimoDia === ayerStr();
  return {
    estrellas: e.estrellas,
    racha: vigente ? e.racha : 0,
    mejorRacha: e.mejorRacha,
    practicoHoy: e.ultimoDia === hoyStr(),
  };
}

/// Registra un intento válido (correcta true/false). Devuelve qué cambió
/// para poder avisarle al chico.
export function registrarIntentoValido(correcta) {
  const e = leer();
  const hoy = hoyStr();
  let rachaNueva = false;

  if (e.ultimoDia !== hoy) {
    e.racha = e.ultimoDia === ayerStr() ? e.racha + 1 : 1;
    e.ultimoDia = hoy;
    rachaNueva = true;
  }
  e.mejorRacha = Math.max(e.mejorRacha, e.racha);

  const estrellaGanada = correcta === true;
  if (estrellaGanada) e.estrellas += 1;

  guardar(e);
  return { estrellaGanada, rachaNueva, racha: e.racha, estrellas: e.estrellas };
}

/// Aviso corto arriba de la pantalla (se saca solo). Se arma directo en el
/// DOM a propósito: así funciona desde cualquier pantalla, incluso mientras
/// está abierta la de práctica con cámara, sin tocar su lógica.
export function mostrarPremio(texto) {
  if (typeof document === "undefined") return;
  const el = document.createElement("div");
  el.className = "premio-toast";
  el.textContent = texto;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add("premio-toast-out"), 2300);
  setTimeout(() => el.remove(), 2800);
}

/// Texto del aviso según lo que pasó, o null si no hay nada que festejar.
export function textoPremio(r) {
  const partes = [];
  if (r.estrellaGanada) partes.push("⭐ ¡+1 estrella!");
  if (r.rachaNueva) {
    partes.push(r.racha > 1 ? `🔥 ¡Racha de ${r.racha} días!` : "🔥 ¡Empezaste tu racha!");
  }
  return partes.length ? partes.join("  ") : null;
}
