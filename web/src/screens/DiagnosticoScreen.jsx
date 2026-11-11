import { useMemo, useState } from "react";
import { PRACTICA_SECCIONES, ORDEN_SONIDOS, SONIDOS_EXTRA } from "../palabras.js";
import { TITULO_SONIDO } from "../progresoSonidos.js";
import { PalabraStage } from "./SonidosScreen.jsx";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";
import "./HomeScreen.css";

const STORAGE_RESULT_KEY = "fonokids_diagnostico";
const STORAGE_DONE_KEY = "fonokids_diagnostico_done";
// Resultado por PALABRA del diagnóstico: { palabra: { sabe: true/false, t: ms } }.
// Se guarda apenas termina cada prueba y no depende de la base de datos.
const STORAGE_PALABRAS_KEY = "fonokids_diagnostico_palabras";

export function getDiagnosticoPalabras() {
  try {
    const raw = localStorage.getItem(STORAGE_PALABRAS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function getDiagnosticoGuardado() {
  try {
    const raw = localStorage.getItem(STORAGE_RESULT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function yaHizoDiagnostico() {
  try {
    return localStorage.getItem(STORAGE_DONE_KEY) === "1";
  } catch {
    return false;
  }
}

// Prueba inicial AL AZAR: palabras fáciles de los sonidos fáciles (las primeras de cada lista
// "al principio"), una por sonido, para ver cuánto puede decir el chico. Cada vez sale otra.
const CUANTAS = 8;
const SONIDOS_FACILES = ORDEN_SONIDOS.slice(0, 11); // m, p, n, t, b, d, f, c-k, g, ch, ll

function faciles() {
  const grupos = PRACTICA_SECCIONES.flatMap((s) => s.grupos);
  const porSonido = {};
  for (const clave of SONIDOS_FACILES) {
    const key = clave === "c-k" ? "pr_k" : `pr_${clave}`;
    const g = grupos.find((x) => x.key === key) || SONIDOS_EXTRA.find((x) => x.key === key);
    const nivel = g?.niveles?.[0]; // "al principio"
    if (!nivel) continue;
    porSonido[clave] = [...nivel.palabras]
      .sort((a, b) => a.palabra.length - b.palabra.length)
      .slice(0, 3); // las más cortas = las más fáciles
  }
  return porSonido;
}

function elegirAlAzar() {
  const pool = faciles();
  const claves = Object.keys(pool).sort(() => Math.random() - 0.5).slice(0, CUANTAS);
  return claves.map((clave) => {
    const lista = pool[clave];
    const p = lista[Math.floor(Math.random() * lista.length)];
    return { sonido: clave, ...p };
  });
}

/// Prueba inicial: palabras fáciles al azar. Dice cuántas pudo y de qué sonidos.
export default function DiagnosticoScreen({ onFinish }) {
  const [fase, setFase] = useState("intro"); // intro | prueba | resultado
  const [modo, setModo] = useState(null);
  const [idx, setIdx] = useState(0);
  const [resultados, setResultados] = useState([]);
  const palabras = useMemo(() => elegirAlAzar(), []);

  function empezar(m) {
    setModo(m);
    setFase("prueba");
  }

  function avanzar(sabe, saltada = false) {
    const actual = palabras[idx];
    const nuevos = [...resultados, { sonido: actual.sonido, palabra: actual.palabra, emoji: actual.emoji, sabe: saltada ? false : sabe, saltada }];
    if (!saltada) {
      try {
        const pal = getDiagnosticoPalabras();
        pal[actual.palabra] = { sabe, t: Date.now() };
        localStorage.setItem(STORAGE_PALABRAS_KEY, JSON.stringify(pal));
        const porSonido = getDiagnosticoGuardado() || {};
        porSonido[actual.sonido] = sabe;
        localStorage.setItem(STORAGE_RESULT_KEY, JSON.stringify(porSonido));
      } catch { /* sin almacenamiento: igual se ve el resumen */ }
    }
    setResultados(nuevos);
    if (idx + 1 < palabras.length) {
      setIdx(idx + 1);
    } else {
      try { localStorage.setItem(STORAGE_DONE_KEY, "1"); } catch { /* nada */ }
      setFase("resultado");
    }
  }

  if (fase === "intro") {
    return (
      <div className="path-screen">
        <div className="path-header">
          <span className="path-title"><span className="path-mascot">🗣️</span> Antes de arrancar...</span>
        </div>
        <div className="mode-choice-stage">
          <div className="mode-choice-title">¡Hola! Vamos a hacer una pequeña prueba</div>
          <div className="repeat-hint" style={{ maxWidth: 420, textAlign: "center" }}>
            Son {CUANTAS} palabras fáciles, elegidas al azar, para ver cuántas puedes decir. Si alguna no te sale,
            la puedes saltar.
          </div>
          <div className="mode-choice-title">¿Cómo quieres hacerla?</div>
          <div className="mode-choice-buttons">
            <button className="mode-choice-button" onClick={() => empezar("camara")}>
              <span className="mode-choice-icon">📷</span>
              Con cámara
            </button>
            <button className="mode-choice-button" onClick={() => empezar("voz")}>
              <span className="mode-choice-icon">🎤</span>
              Solo con la voz
            </button>
          </div>
          <button
            className="repeat-hint"
            style={{ background: "none", border: "none", cursor: "pointer", fontWeight: 700 }}
            onClick={() => {
              try { localStorage.setItem(STORAGE_DONE_KEY, "1"); } catch { /* nada */ }
              onFinish && onFinish(getDiagnosticoGuardado());
            }}
          >
            Prefiero saltear esto
          </button>
        </div>
      </div>
    );
  }

  if (fase === "prueba") {
    const actual = palabras[idx];
    return (
      <PalabraStage
        key={`${idx}-${actual.palabra}`}
        item={actual}
        claveGuardado={`test_${actual.sonido}_${actual.palabra}`}
        tipo="diagnostico"
        modoEvaluacion
        modoInicial={modo}
        indice={idx}
        total={palabras.length}
        onTerminar={(bien) => avanzar(bien)}
        onBack={() => avanzar(false, true)}
      />
    );
  }

  const evaluados = resultados.filter((r) => !r.saltada);
  const bien = evaluados.filter((r) => r.sabe).length;
  const saltadas = resultados.length - evaluados.length;
  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🗣️</span> ¡Terminamos!</span>
      </div>
      <div className="mode-choice-stage resultado-stage">
        <div className="mode-choice-title">
          {evaluados.length === 0
            ? "¡Sin problema! Después las hacemos 😊"
            : `Dijiste bien ${bien} de ${evaluados.length} palabras ${bien === evaluados.length ? "🌟" : "👏"}`}
        </div>
        <div className="repeat-hint" style={{ textAlign: "center" }}>
          {saltadas > 0 ? `Saltaste ${saltadas}. ` : ""}Así armo tu camino; los sonidos que costaron salen para practicar.
        </div>
        <div className="etapa-list resultado-lista" style={{ width: "100%", maxWidth: 480 }}>
          {resultados.map((r) => (
            <div
              key={r.sonido}
              className={`etapa-card etapa-color-${r.saltada ? 0 : r.sabe ? 1 : 3}`}
              style={{ cursor: "default" }}
            >
              <span className="etapa-card-word">
                <span className="etapa-card-icon">{r.emoji}</span> {r.palabra}
                <span className="etapa-card-tag" style={{ marginLeft: 8 }}>{TITULO_SONIDO[r.sonido] || r.sonido}</span>
              </span>
              <span className="etapa-card-action">
                {r.saltada ? "Saltada ⏭️" : r.sabe ? "¡Bien! ✅" : "Para practicar 🔁"}
              </span>
            </div>
          ))}
        </div>
        <button className="path-cta" onClick={() => onFinish && onFinish(getDiagnosticoGuardado())}>
          Vamos a practicar →
        </button>
      </div>
    </div>
  );
}
