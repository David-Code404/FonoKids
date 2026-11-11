import test from "node:test";
import assert from "node:assert/strict";

const mem = {};
globalThis.localStorage = { getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); } };
globalThis.window = { location: { hostname: "localhost" } };
const f = await import("../src/fonemas.js");
const ps = await import("../src/progresoSonidos.js");
const pal = await import("../src/palabras.js");

const partir = (w) => f.fonemasDe(w).map((s) => s.letras).join("-");

test("separa en fonemas: dígrafos, QU, GU, C suave, H muda", () => {
  assert.equal(partir("rana"), "R-A-N-A");
  assert.equal(partir("queso"), "QU-E-S-O");
  assert.equal(partir("chuleta"), "CH-U-L-E-T-A");
  assert.equal(partir("guitarra"), "GU-I-T-A-RR-A");
  assert.equal(partir("pollo"), "P-O-LL-O");
  assert.equal(partir("hormiga"), "O-R-M-I-G-A");
  assert.equal(partir("cielo"), "C-I-E-L-O");
});

test("TODAS las palabras de la app se pueden partir y cada sonido tiene forma y texto", () => {
  let n = 0;
  for (const s of pal.PRACTICA_SECCIONES) for (const g of s.grupos) for (const nv of g.niveles) for (const p of nv.palabras) {
    const u = f.fonemasDe(p.palabra);
    assert.ok(u.length >= 2, `muy corta: ${p.palabra}`);
    for (const x of u) {
      assert.ok(x.letras && x.sonido, `unidad vacía en ${p.palabra}`);
      assert.ok(["circulo", "triangulo", "cuadrado"].includes(f.formaDe(x)));
      assert.ok(f.comoSeVe(x));
    }
    n++;
  }
  assert.ok(n > 150);
});

test("videos: cada vocal y cada sonido de las consonantes principales tienen video", () => {
  for (const k of ["a", "e", "i", "o", "u", "m", "p", "n", "t", "b", "d", "f", "k", "g", "ch", "ll", "y", "ñ", "l", "s", "r", "rr", "j", "z"]) {
    assert.ok(f.videosDe(k).length >= 1, `sin video: ${k}`);
  }
});

test("los ids guardados se leen bien en Logros / Para practicar", () => {
  const c = (id) => JSON.stringify(ps.parsearIdPalabra(id));
  assert.equal(c("fon_pr_m_mano"), '{"sonido":"m","palabra":"mano"}');
  assert.equal(c("pr_r_1_naranja"), '{"sonido":"rr","palabra":"naranja"}');
  assert.equal(c("test_c-k_casa"), '{"sonido":"c-k","palabra":"casa"}');
  assert.equal(c("r_rana"), '{"sonido":"rr","palabra":"rana"}');
  assert.equal(c("sinfonesR_tren"), '{"sonido":"sinfones","palabra":"tren"}');
  assert.equal(ps.parsearIdPalabra("cosa_rara"), null);
});

test("unirConPalabrasNuevas cuenta bien y mal de Ejercicios, pruebas y evaluaciones", () => {
  const resumen = [
    { tipo: "palabra", ejercicio_id: "fon_pr_m_mano", intentos: 3, bien: 2, ultimo: "2026-11-10" },
    { tipo: "diagnostico", ejercicio_id: "test_p_pan", intentos: 1, bien: 0, ultimo: "2026-11-10" },
    { tipo: "evaluacion", ejercicio_id: "s_sol", intentos: 2, bien: 2, ultimo: "2026-11-10" },
    { tipo: "praxia", ejercicio_id: "sonrisa", intentos: 5, bien: 5, ultimo: "2026-11-10" },
  ];
  const l = ps.unirConPalabrasNuevas([], resumen);
  assert.equal(l.length, 4); // mano bien, mano mal, pan mal, sol bien (la praxia no entra)
  assert.equal(l.filter((x) => x.correcta).reduce((a, x) => a + x.count, 0), 4);
  assert.equal(l.filter((x) => !x.correcta).reduce((a, x) => a + x.count, 0), 2);
  const g = ps.agruparPorSonido(l);
  assert.deepEqual(g.map((x) => x.clave), ["m", "p", "s"]);
});

test("la lista de Ejercicios no repite sonidos (la L salía dos veces)", async () => {
  const claves = [];
  const vistos = new Set();
  for (const s of pal.PRACTICA_SECCIONES) for (const g of s.grupos) if (g.key.startsWith("pr_") && !vistos.has(g.key)) { vistos.add(g.key); claves.push(g.key); }
  assert.equal(new Set(claves).size, claves.length);
  assert.ok(claves.includes("pr_l"));
});
