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

test("por palabras: intermedia a las 5 logradas desde la previa y final al lograr todas", () => {
  const lista = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"];
  for (const k of Object.keys(mem)) delete mem[k];
  let est = p.estadoEvaluacion("zz", lista, 5);
  assert.equal(est.previa.toca, true);
  assert.equal(est.intermedia.toca, false);
  p.guardarEvaluacion("zz", "previa", 1, 3, p.contarLogradas(lista));
  ["a1", "a2", "a3", "a4"].forEach(p.marcarLograda);
  est = p.estadoEvaluacion("zz", lista, 5);
  assert.equal(est.intermedia.toca, false);
  assert.equal(est.intermedia.faltan, 1);
  p.marcarLograda("a5");
  assert.equal(p.estadoEvaluacion("zz", lista, 5).intermedia.toca, true);
  p.guardarEvaluacion("zz", "intermedia", 2, 3, p.contarLogradas(lista));
  assert.equal(p.estadoEvaluacion("zz", lista, 5).final.toca, false);
  p.marcarLograda("a6"); p.marcarLograda("a7");
  assert.equal(p.estadoEvaluacion("zz", lista, 5).final.toca, true);
  p.guardarEvaluacion("zz", "final", 3, 3, 7);
  assert.equal(p.estadoEvaluacion("zz", lista, 5).final.toca, false);
});

test("intermedia y final se abren por palabras logradas aunque no se haya hecho el test previo", () => {
  const lista = ["b1", "b2", "b3", "b4", "b5", "b6"];
  for (const k of Object.keys(mem)) delete mem[k];
  lista.slice(0, 5).forEach(p.marcarLograda);
  let est = p.estadoEvaluacion("yy", lista, 5);
  assert.equal(est.intermedia.toca, true);
  assert.equal(est.final.toca, false);
  p.marcarLograda("b6");
  est = p.estadoEvaluacion("yy", lista, 5);
  assert.equal(est.final.toca, true);
});

test("reiniciarSonido borra las palabras logradas y las pruebas intermedia y final, pero deja la previa", () => {
  for (const k of Object.keys(mem)) delete mem[k];
  const lista = ["c1", "c2", "c3"];
  lista.forEach(p.marcarLograda);
  p.guardarEvaluacion("xx", "previa", 2, 3);
  p.guardarEvaluacion("xx", "intermedia", 1, 3);
  p.reiniciarSonido("xx", lista);
  assert.equal(p.contarLogradas(lista), 0);
  assert.equal(p.evaluaciones().xx.intermedia, undefined);
  assert.equal(p.evaluaciones().xx.previa.bien, 2);
});

test("una prueba con menos de 2 estrellas se descarta y reinicia el sonido", () => {
  for (const k of Object.keys(mem)) delete mem[k];
  const lista = ["d1", "d2", "d3", "d4", "d5", "d6"];
  p.guardarEvaluacion("ww", "previa", 1, 3, 0);
  lista.slice(0, 5).forEach(p.marcarLograda);
  p.guardarEvaluacion("ww", "intermedia", 1, 3, 5);
  const est = p.estadoEvaluacion("ww", lista, 5);
  assert.equal(est.logradas, 0);
  assert.equal(est.intermedia.hecha, null);
  assert.equal(est.intermedia.toca, false);
  assert.equal(est.e.previa.bien, 1);
});
