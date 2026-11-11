import test from "node:test";
import assert from "node:assert/strict";

const mem = {};
globalThis.localStorage = {
  getItem: (k) => (k in mem ? mem[k] : null),
  setItem: (k, v) => { mem[k] = String(v); },
};
const p = await import("../src/progreso.js");

test("la evaluación previa toca primero; intermedia y final cada 5 ejercicios", () => {
  assert.equal(p.evaluacionQueToca("r", 5), "previa");
  p.guardarEvaluacion("r", "previa", 1, 3);
  assert.equal(p.evaluacionQueToca("r", 5), null);
  assert.equal(p.ejerciciosParaSiguiente("r", 5), 5);
  for (let i = 0; i < 5; i++) p.registrarEjercicioHecho();
  assert.equal(p.evaluacionQueToca("r", 5), "intermedia");
  p.guardarEvaluacion("r", "intermedia", 2, 3);
  assert.equal(p.evaluacionQueToca("r", 5), null);
  for (let i = 0; i < 5; i++) p.registrarEjercicioHecho();
  assert.equal(p.evaluacionQueToca("r", 5), "final");
  p.guardarEvaluacion("r", "final", 3, 3);
  assert.equal(p.evaluacionQueToca("r", 5), null);
  assert.equal(p.evaluaciones().r.final.bien, 3);
});

test("los días de uso no se repiten y las palabras logradas se guardan", () => {
  assert.equal(p.diasDeUso(), 1);
  p.marcarLograda("sal");
  assert.equal(p.palabrasLogradas().sal, true);
});
