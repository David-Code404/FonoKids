// Pruebas de la estimación del esfuerzo (web/src/ejercicios.js). Se corren con
// `npm test`. Usan señales sintéticas: no necesitan cámara ni micrófono.
import test from "node:test";
import assert from "node:assert/strict";
import { esfuerzoPraxia, esfuerzoSoplo, esfuerzoVoz, ETIQUETAS_ESFUERZO } from "../src/ejercicios.js";

const rep = (fuerza, firme) => ({ fuerza, firme });

// ---- bloques de audio sintéticos (4096 muestras, como el grabador) ----
function bloque(amplitud, n = 4096) {
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) b[i] = amplitud * Math.sin(i * 0.3) * (0.7 + 0.3 * Math.sin(i * 0.011));
  return b;
}
const audio = (ruido, voz, nRuido, nVoz) => [...Array(nRuido).fill(0).map(() => bloque(ruido)), ...Array(nVoz).fill(0).map(() => bloque(voz))];

test("praxia: gesto fuerte, firme y parejo = 3 estrellas en todo", () => {
  assert.deepEqual(esfuerzoPraxia([rep(0.95, 0.95), rep(0.93, 0.9), rep(0.96, 0.97)]), { fuerza: 3, aguante: 3, parejo: 3 });
});

test("praxia: gesto justo (apenas sobre el umbral) y desparejo = pocas estrellas", () => {
  const r = esfuerzoPraxia([rep(0.55, 0.1), rep(0.8, 0.5), rep(0.6, 0.2)]);
  assert.equal(r.aguante, 1);
  assert.equal(r.parejo, 1);
  assert.ok(r.fuerza <= 2);
});

test("praxia: tres repeticiones idénticas son perfectamente parejas", () => {
  assert.equal(esfuerzoPraxia([rep(0.7, 0.6), rep(0.7, 0.6), rep(0.7, 0.6)]).parejo, 3);
});

test("praxia: más fuerza nunca baja las estrellas (monotonía)", () => {
  let previo = 0;
  for (let f = 0.5; f <= 1.2; f += 0.05) {
    const e = esfuerzoPraxia([rep(f, 0.6), rep(f, 0.6), rep(f, 0.6)]).fuerza;
    assert.ok(e >= previo, `fuerza ${f.toFixed(2)} bajó de ${previo} a ${e}`);
    previo = e;
  }
  assert.equal(previo, 3);
});

test("praxia: sin repeticiones no inventa resultados", () => {
  assert.equal(esfuerzoPraxia([]), null);
});

test("praxia: siempre entrega niveles enteros de 1 a 3", () => {
  for (const f of [0, 0.3, 0.65, 0.9, 1.5]) {
    for (const g of [0, 0.5, 1]) {
      for (const v of Object.values(esfuerzoPraxia([rep(f, g), rep(f * 0.5, g), rep(f, g)]))) {
        assert.ok([1, 2, 3].includes(v), `valor fuera de rango: ${v}`);
      }
    }
  }
});

test("soplo: fuerte y sostenido vs. flojo y cortado", () => {
  assert.deepEqual(esfuerzoSoplo(0.2, 0.06, 0.95), { fuerza: 3, aguante: 3 });
  assert.deepEqual(esfuerzoSoplo(0.07, 0.06, 0.3), { fuerza: 1, aguante: 1 });
});

test("soplo: justo sobre el umbral no da la máxima fuerza", () => {
  assert.ok(esfuerzoSoplo(0.061, 0.06, 0.9).fuerza < 3);
});

test("voz: habla clara sobre ruido bajo = 3 estrellas", () => {
  assert.equal(esfuerzoVoz(audio(0.004, 0.2, 4, 8)).voz, 3);
});

test("voz: voz apenas por encima del ruido = 1 estrella", () => {
  assert.equal(esfuerzoVoz(audio(0.05, 0.07, 6, 6)).voz, 1);
});

test("voz: casi silencio = 1 estrella", () => {
  assert.equal(esfuerzoVoz(audio(0.0005, 0.001, 6, 6)).voz, 1);
});

test("voz: voz más fuerte nunca baja las estrellas (monotonía)", () => {
  let previo = 0;
  for (const v of [0.01, 0.02, 0.04, 0.08, 0.16, 0.3]) {
    const e = esfuerzoVoz(audio(0.005, v, 4, 8)).voz;
    assert.ok(e >= previo, `voz ${v} bajó de ${previo} a ${e}`);
    previo = e;
  }
});

test("voz: audio demasiado corto o vacío no se mide", () => {
  assert.equal(esfuerzoVoz([]), null);
  assert.equal(esfuerzoVoz(null), null);
  assert.equal(esfuerzoVoz([bloque(0.2), bloque(0.2)]), null);
});

test("las etiquetas cubren todos los indicadores y no hay porcentajes", () => {
  for (const k of ["fuerza", "aguante", "parejo", "voz"]) assert.ok(ETIQUETAS_ESFUERZO[k], `falta etiqueta de ${k}`);
  for (const t of Object.values(ETIQUETAS_ESFUERZO)) assert.ok(!t.includes("%"));
});
