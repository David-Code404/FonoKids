// Arma la partida guiada del modo niños: Fofo elige la letra y las palabras; el niño solo juega.
import { PRACTICA_SECCIONES, palabrasDeGrupo, evaluacionDeGrupo } from "./palabras.js";
import { palabrasLogradas, evaluaciones, contarLogradas } from "./progreso.js";

// De los sonidos más fáciles a los más difíciles.
const ORDEN = ["pr_m", "pr_p", "pr_n", "pr_t", "pr_b", "pr_d", "pr_f", "pr_k", "pr_g", "pr_ch", "pr_ll", "pr_y", "pr_ni", "pr_l", "pr_j", "pr_z", "pr_s", "pr_r", "sf_l", "sf_r"];
export const PALABRAS_POR_PARTIDA = 5;

function grupo(clave) {
  for (const sec of PRACTICA_SECCIONES) for (const g of sec.grupos) if (g.key === clave) return g;
  return null;
}

function palabrasDe(g) {
  const vistas = new Set();
  const lista = [];
  for (const nv of g.niveles) for (const p of nv.palabras) if (!vistas.has(p.palabra)) { vistas.add(p.palabra); lista.push(p); }
  return lista;
}

// ---------------------------------------------------------------- TEST PREVIO (una sola vez, al empezar)
// Una palabra por cada sonido que a los 6 o 7 años todavía suele costar. Los sonidos tempranos (M, P, N, T, B, F)
// se dan por dominados a esta edad. Lo que sale mal en el test define lo que Fofo hace practicar.
const K_TEST = "fk_test_previo";
const GRUPOS_TEST = ["pr_d", "pr_k", "pr_g", "pr_ch", "pr_ll", "pr_l", "pr_j", "pr_s", "pr_r", "sf_l", "sf_r"];

function leerTest() {
  try { return JSON.parse(localStorage.getItem(K_TEST) || "null"); } catch { return null; }
}
export const testPrevioHecho = () => !!leerTest()?.hecho;
export const resultadoTestPrevio = () => leerTest()?.grupos || {};
export function guardarTestPrevio(grupos) {
  try { localStorage.setItem(K_TEST, JSON.stringify({ hecho: true, t: Date.now(), grupos })); } catch { /* nada */ }
}
// El nombre del niño (se pide una sola vez, para que Fofo lo salude).
const K_NOMBRE = "fk_nombre";
export function getNombre() {
  try { return localStorage.getItem(K_NOMBRE) || ""; } catch { return ""; }
}
export function setNombre(n) {
  try { localStorage.setItem(K_NOMBRE, (n || "").trim().slice(0, 14)); } catch { /* nada */ }
}

/// Los círculos del camino: un círculo por cada sonido del test previo. estado: true = lo dijo bien, false = a practicar.
export function gruposDelCamino() {
  const res = resultadoTestPrevio();
  return ORDEN.map(grupo).filter((g) => g && g.key in res).map((g) => ({ clave: g.key, titulo: g.titulo, icono: g.icono, estado: res[g.key] }));
}

export function reiniciarTestPrevio() {
  try { localStorage.removeItem(K_TEST); } catch { /* nada */ }
}

/// Las palabras del test previo: [{ palabra, emoji, grupo, titulo, icono }]
export function palabrasTestPrevio() {
  const lista = [];
  for (const clave of GRUPOS_TEST) {
    const g = grupo(clave);
    const ev = g && evaluacionDeGrupo(clave);
    if (!g || !ev) continue;
    const elegidas = clave === "pr_r" ? ev.palabras.slice(0, 2) : [ev.palabras.find((p) => p.pos === "inicial") || ev.palabras[0]];
    for (const p of elegidas) lista.push({ ...p, grupo: clave, titulo: g.titulo, icono: g.icono });
  }
  return lista;
}

// Estado de un sonido en el camino: qué prueba toca (del medio o final) o si ya terminó.
//  - intermedia: al lograr 5 palabras; final: al lograr TODAS las palabras del sonido.
function estadoSonido(g) {
  const nombres = palabrasDeGrupo(g.key);
  const ev = evaluacionDeGrupo(g.key);
  const e = (ev && evaluaciones()[ev.key]) || {};
  const logradas = contarLogradas(nombres);
  const total = nombres.length;
  let toca = null;
  if (ev) {
    if (!e.intermedia && logradas >= PALABRAS_POR_PARTIDA) toca = "intermedia";
    else if (!e.final && e.intermedia && logradas >= total) toca = "final";
  }
  const terminado = (!ev || !!e.final) && logradas >= total;
  return { ev, toca, terminado };
}

/// La próxima partida. Primero el TEST PREVIO (una sola vez). Después, solo los sonidos que salieron mal en el test,
/// de los más fáciles a los más difíciles: práctica de 5 palabras, prueba del medio y reto final.
/// Devuelve { tipo: "testprevio" | "prueba" | "practica", ... } o null si ya aprendió todo.
export function armarPartida(claveForzada = null) {
  if (!testPrevioHecho()) {
    const palabras = palabrasTestPrevio();
    return { tipo: "testprevio", clave: "testprevio", titulo: "Juego de Fofo", icono: "🎮", sonido: null, palabras };
  }
  const logradas = palabrasLogradas();
  const res = resultadoTestPrevio();
  if (claveForzada) {
    // El niño tocó un círculo del camino: se juega ese sonido (con sus pruebas si le toca y le costaba).
    const g = grupo(claveForzada);
    if (g) {
      const st = estadoSonido(g);
      const base = { clave: g.key, titulo: g.titulo, icono: g.icono, sonido: g.key.startsWith("pr_") ? g.key.slice(3) : null };
      if (st.toca && res[g.key] === false) return { ...base, tipo: "prueba", etapa: st.toca, evalDef: st.ev, palabras: st.ev.palabras };
      const todas = palabrasDe(g);
      const faltan = todas.filter((p) => !logradas[p.palabra]);
      return { ...base, tipo: "practica", palabras: (faltan.length ? faltan : todas).slice(0, PALABRAS_POR_PARTIDA) };
    }
  }
  const grupos = ORDEN.map(grupo).filter((g) => g && res[g.key] === false);
  for (const g of grupos) {
    const st = estadoSonido(g);
    if (st.terminado) continue;
    const base = {
      clave: g.key, titulo: g.titulo, icono: g.icono,
      sonido: g.key.startsWith("pr_") ? g.key.slice(3) : null,
    };
    if (st.toca) return { ...base, tipo: "prueba", etapa: st.toca, evalDef: st.ev, palabras: st.ev.palabras };
    const faltan = palabrasDe(g).filter((p) => !logradas[p.palabra]);
    if (faltan.length) return { ...base, tipo: "practica", palabras: faltan.slice(0, PALABRAS_POR_PARTIDA) };
  }
  return null;
}

export function palabrasAprendidas() {
  const l = palabrasLogradas();
  const lista = [];
  for (const sec of PRACTICA_SECCIONES) for (const g of sec.grupos) for (const p of palabrasDe(g)) {
    if (l[p.palabra] && !lista.some((x) => x.palabra === p.palabra)) lista.push(p);
  }
  return lista;
}

// ---------------------------------------------------------------- CUENTA DEL NIÑO (se guarda en la base de datos)
// Al escribir su nombre se busca en la base: si ya había jugado, se recupera su avance (test previo, logros, rachas...) y no tiene
// que repetir nada. Mientras juega, su avance se va guardando solo.
import { getServerUrl } from "./api.js";

const K_NINO_ID = "fk_nino_id";
// Lo que NO es del niño sino de este aparato: dirección del servidor, voz, cámara, nombre e id.
const NO_SINCRONIZAR = new Set(["speakshadow_server_url", "fonokids_voz", "fk_usar_camara", "fk_nombre", K_NINO_ID]);
const guardarLocal = Storage.prototype.setItem;
let temporizador = null;
let sincronizando = false;

const sincronizable = (k) => !NO_SINCRONIZAR.has(k) && (k.startsWith("fk_") || k.startsWith("fonokids_"));

function instantanea() {
  const estado = {};
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const k = localStorage.key(i);
      if (sincronizable(k)) estado[k] = localStorage.getItem(k);
    }
  } catch { /* nada */ }
  return estado;
}

async function subirAvance() {
  let id = null;
  try { id = localStorage.getItem(K_NINO_ID); } catch { /* nada */ }
  if (!id) return;
  try {
    await fetch(`${getServerUrl()}/ninos/${id}/estado`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ estado: instantanea() }),
    });
  } catch { /* sin servidor: queda en este aparato */ }
}

function activarGuardadoAutomatico() {
  if (sincronizando) return;
  sincronizando = true;
  Storage.prototype.setItem = function (k, v) {
    guardarLocal.call(this, k, v);
    if (this === window.localStorage && sincronizable(k)) {
      clearTimeout(temporizador);
      temporizador = setTimeout(subirAvance, 1500);
    }
  };
}
try { if (localStorage.getItem(K_NINO_ID)) activarGuardadoAutomatico(); } catch { /* nada */ }

/// Entra el niño por su nombre. Devuelve { nuevo } (nuevo = primera vez que entra, o no hay base de datos).
export async function entrarNino(nombre) {
  try {
    const r = await fetch(`${getServerUrl()}/ninos/entrar`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nombre: (nombre || "").trim() }),
    });
    const j = await r.json();
    if (!j || !j.id) return { nuevo: true };
    // Limpia lo que hubiera de otro niño en este aparato y deja lo de este.
    const viejas = instantanea();
    for (const k of Object.keys(viejas)) { try { localStorage.removeItem(k); } catch { /* nada */ } }
    if (!j.nuevo && j.estado) {
      try {
        const estado = JSON.parse(j.estado);
        for (const [k, v] of Object.entries(estado)) if (sincronizable(k)) guardarLocal.call(localStorage, k, v);
      } catch { /* estado dañado: empieza de cero */ }
    }
    guardarLocal.call(localStorage, K_NINO_ID, String(j.id));
    activarGuardadoAutomatico();
    return { nuevo: !!j.nuevo };
  } catch {
    return { nuevo: true };
  }
}
