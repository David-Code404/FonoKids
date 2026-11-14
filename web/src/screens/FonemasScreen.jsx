import { useEffect, useMemo, useRef, useState } from "react";
import { getServerUrl, ejercicioVoz } from "../api.js";
import { createAudioRecorder } from "../audioRecorder.js";
import { PalabraStage, EvaluacionRun } from "./SonidosScreen.jsx";
import { PRACTICA_SECCIONES, ORDEN_SONIDOS, EJERCICIOS_POR_EVALUACION, evaluacionDeGrupo } from "../palabras.js";
import { fonemasDe, decirSonido, decirEnVoz, esVocal, comoSeVe, formaDe, videosDe, ejemploDe } from "../fonemas.js";
import { sonidoBien, sonidoFalta } from "../sonidos.js";
import { palabrasLogradas, estadoEvaluacion } from "../progreso.js";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";
import "./HomeScreen.css";
import "./SonidosScreen2.css";
import "./FonemasScreen.css";

// Un grupo por sonido (R, S, L, ...) con sus palabras sin repetir, de los sonidos más fáciles
// a los más difíciles.
function sonidosConPalabras() {
  // La L está en dos secciones de palabras.js (la principal y "Más sonidos"): se cuenta una sola vez.
  const vistos = new Set();
  const grupos = PRACTICA_SECCIONES.flatMap((s) => s.grupos)
    .filter((g) => g.key.startsWith("pr_") && !vistos.has(g.key) && vistos.add(g.key));
  const orden = (g) => {
    const clave = g.key.slice(3);
    const i = ORDEN_SONIDOS.findIndex((o) => o === clave || (clave === "r" && o === "rr") || (clave === "k" && o === "c-k"));
    return i < 0 ? 99 : i;
  };
  return grupos
    .map((g) => {
      const vistas = new Set();
      const palabras = g.niveles.flatMap((n) => n.palabras).filter((p) => !vistas.has(p.palabra) && vistas.add(p.palabra));
      return { key: g.key, titulo: g.titulo, icono: g.icono, palabras };
    })
    .sort((a, b) => orden(a) - orden(b));
}

function Encabezado({ icono, titulo, onBack }) {
  return (
    <div className="path-header">
      {onBack && <button className="icon-button" onClick={onBack}>←</button>}
      <span className="path-title" style={{ textTransform: "none" }}>{icono} {titulo}</span>
    </div>
  );
}


// El niño graba el sonido; si sale bien pasa solo al siguiente. Las consonantes se miden con el
// detector (sonido suelto: tolerancia mayor que en palabras); las vocales solo se revisan de
// forma muy suave porque el detector no las distingue bien sueltas. Tras 3 intentos deja seguir.
const UMBRAL_CONSONANTE = -2.5;
const UMBRAL_VOCAL = -6.0;

export function GrabarSonido({ sonido, onBien }) {
  const streamRef = useRef(null);
  const recRef = useRef(createAudioRecorder());
  const [listo, setListo] = useState(false);
  const [grabando, setGrabando] = useState(false);
  const [procesando, setProcesando] = useState(false);
  const [aviso, setAviso] = useState("");
  const [fallos, setFallos] = useState(0);

  useEffect(() => {
    let cancelado = false;
    navigator.mediaDevices.getUserMedia({ audio: true }).then((st) => {
      if (cancelado) { st.getTracks().forEach((t) => t.stop()); return; }
      streamRef.current = st;
      setListo(true);
    }).catch((e) => setAviso(`No se pudo usar el micrófono: ${e.message || e}`));
    return () => { cancelado = true; streamRef.current?.getTracks().forEach((t) => t.stop()); };
  }, []);

  function empezar() {
    if (streamRef.current && recRef.current.start(streamRef.current)) { setGrabando(true); setAviso(""); }
  }
  async function terminar() {
    setGrabando(false);
    const blob = recRef.current.stop();
    if (!blob) return;
    setProcesando(true);
    try {
      const r = await ejercicioVoz(getServerUrl(), blob, sonido.sonido, "sonido");
      if (!r.valid) { setAviso(r.reason || "No te escuché, ¡prueba otra vez!"); return; }
      // El servidor decide (el sonido pedido tiene que aparecer en lo que reconoce); el corte por puntaje
      // queda solo como respaldo si el servidor no manda veredicto.
      const score = r.scores?.[0];
      const umbral = esVocal(sonido.letras) ? UMBRAL_VOCAL : UMBRAL_CONSONANTE;
      const bien = typeof r.correcta === "boolean" ? r.correcta : (score == null || score >= umbral);
      if (bien) { sonidoBien(); onBien(); }
      else { sonidoFalta(); setFallos((n) => n + 1); setAviso("¡Casi! Escucha otra vez y prueba de nuevo"); }
    } catch (e) {
      setAviso(e.message || "Error al analizar el audio.");
    } finally {
      setProcesando(false);
    }
  }

  return (
    <>
      {listo && !procesando && (
        <button className="path-cta" onClick={grabando ? terminar : empezar}>
          {grabando ? "Listo →" : "🎤 Decir el sonido"}
        </button>
      )}
      {procesando && <div className="audio-only-label">Analizando...</div>}
      {aviso && <div className="audio-only-error">{aviso}</div>}
      {!listo && aviso && (
        <button className="cambiar-modo" onClick={onBien}>Seguir sin micrófono →</button>
      )}
      {fallos >= 3 && !grabando && !procesando && (
        <button className="cambiar-modo" onClick={onBien}>Seguir igual →</button>
      )}
    </>
  );
}

// Paso 1: el niño dice cada sonido por separado (sin evaluar). Paso 2: dice la palabra entera.
// Carta del sonido con imagen: arriba el dibujo de una palabra que lleva ese sonido (como en las tarjetas
// de aprender a leer) y abajo la letra grande. El color distingue el tipo de sonido.
function Carta({ forma, texto, estado, ejemplo }) {
  return (
    <div className={`carta carta-${forma} ${estado}`}>
      {ejemplo && <span className="carta-dibujo" aria-hidden="true">{ejemplo.dibujo}</span>}
      <span className="carta-texto">{texto}</span>
      {ejemplo && estado === "activa" && <span className="carta-palabra">{ejemplo.palabra}</span>}
    </div>
  );
}

function PalabraPorSonidos({ item, clave, onBack, onTerminar }) {
  const sonidos = useMemo(() => fonemasDe(item.palabra), [item.palabra]);
  const [i, setI] = useState(0);
  const videoRef = useRef(null);

  // Al salir cada sonido, la app lo dice primero para que el niño lo copie.
  useEffect(() => {
    // Con video de referencia, el video ya suena solo; sin video, la app lo dice con su audio.
    if (i < sonidos.length && !videosDe(sonidos[i].sonido).length) {
      const t = setTimeout(() => decirSonido(sonidos[i].sonido), 300);
      return () => clearTimeout(t);
    }
    return undefined;
  }, [i, sonidos]);

  if (i >= sonidos.length) {
    return (
      <PalabraStage
        key={item.palabra}
        item={item}
        claveGuardado={clave}
        tipo="palabra"
        onTerminar={onTerminar}
        onBack={onBack}
      />
    );
  }
  return (
    <div className="path-screen">
      <Encabezado icono={item.emoji} titulo={`Sonido ${i + 1} de ${sonidos.length}`} onBack={onBack} />
      <div className="repeat-stage ejercicio-stage">
        <div className="fon-cabecera">
          <span className="fon-dibujo">{item.emoji}</span>
          <span className="fon-pista">Escucha y di solo este sonido</span>
        </div>
        <div className="cartas-fila">
          {sonidos.map((u, k) => (
            <Carta key={k} forma={formaDe(u)} ejemplo={ejemploDe(u.sonido)} texto={k === i ? comoSeVe(u) : u.letras}
              estado={k === i ? "activa" : k < i ? "hecha" : ""} />
          ))}
        </div>
        {videosDe(sonidos[i].sonido)[0] ? (
          <>
            <video
              key={`${item.palabra}-${i}`} ref={videoRef} className="como-video fon-video"
              src={videosDe(sonidos[i].sonido)[0]} autoPlay playsInline
            />
            <button className="cambiar-modo" onClick={() => { const v = videoRef.current; if (v) { v.currentTime = 0; v.play(); } }}>
              🔁 Ver otra vez
            </button>
          </>
        ) : (
          <button className="cambiar-modo" onClick={() => decirSonido(sonidos[i].sonido)}>🔁 Escuchar otra vez</button>
        )}
        <GrabarSonido key={i} sonido={sonidos[i]} onBien={() => setI(i + 1)} />
      </div>
    </div>
  );
}


// Pruebas de CADA sonido: previa al empezar, intermedia al lograr 5 palabras y final al lograr todas.
function PruebasDelSonido({ grupo, onAbrir, solo }) {
  const def = evaluacionDeGrupo(grupo.key);
  if (!def) return null;
  const est = estadoEvaluacion(def.key, grupo.palabras.map((p) => p.palabra), EJERCICIOS_POR_EVALUACION);
  const filas = [
    { etapa: "previa", nombre: "Para empezar: ¿cómo te sale?", texto: est.previa.toca ? "Haz esta prueba antes de empezar →" : null },
    {
      etapa: "intermedia", nombre: "¿Cómo vas? Mira cuánto mejoraste",
      texto: est.intermedia.toca ? "¡Ya lograste 5 palabras! Toca hacerla →"
        : `Se abre al lograr ${est.intermedia.faltan} palabra${est.intermedia.faltan === 1 ? "" : "s"} más`,
    },
    {
      etapa: "final", nombre: "¡Gran reto final!",
      texto: est.final.toca ? "¡Lograste todas las palabras! Toca hacerla →"
        : `Se abre al lograr todas las palabras (faltan ${est.final.faltan})`,
    },
  ];
  return (
    <div style={{ padding: "0 14px 8px", display: "flex", flexDirection: "column", gap: 6 }}>
      {filas.map((f, i) => {
        if (solo && f.etapa !== solo) return null;
        const hecha = est[f.etapa].hecha;
        const abierta = est[f.etapa].toca;
        return (
          <button
            key={f.etapa}
            className={`paso-camino paso-eval ${hecha ? "paso-hecho" : ""}`}
            disabled={!abierta}
            onClick={() => onAbrir(def, f.etapa)}
          >
            <span className="paso-num">{i + 1}</span>
            <span className="paso-texto">
              <strong>📋 {f.nombre}</strong>
              <small>{hecha ? `${"⭐".repeat(hecha.bien)}${"☆".repeat(Math.max(hecha.total - hecha.bien, 0))}  ${hecha.bien} de ${hecha.total}` : f.texto}</small>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default function FonemasScreen({ grupoInicial = null, onSalir = null }) {
  const grupos = useMemo(sonidosConPalabras, []);
  const [grupo, setGrupo] = useState(() => grupos.find((g) => g.key === grupoInicial) || null);
  const [palabra, setPalabra] = useState(null);
  const [prueba, setPrueba] = useState(null); // { def, etapa }
  const [, refrescar] = useState(0);

  const abrirSiguiente = () => {
    const lista = grupo.palabras;
    const k = lista.findIndex((p) => p.palabra === palabra.palabra);
    refrescar((n) => n + 1);
    setPalabra(k >= 0 && k + 1 < lista.length ? lista[k + 1] : null);
  };

  if (grupo && prueba) {
    return (
      <EvaluacionRun
        evalDef={prueba.def}
        etapa={prueba.etapa}
        onSalir={() => { setPrueba(null); refrescar((n) => n + 1); }}
      />
    );
  }

  if (grupo && palabra) {
    return (
      <PalabraPorSonidos
        key={palabra.palabra}
        item={palabra}
        clave={`fon_${grupo.key}_${palabra.palabra}`}
        onBack={() => setPalabra(null)}
        onTerminar={abrirSiguiente}
      />
    );
  }

  if (grupo) {
    const logradas = palabrasLogradas();
    return (
      <div className="path-screen">
        <Encabezado icono={grupo.icono} titulo={`Sonido ${grupo.titulo}`} onBack={onSalir ? onSalir : () => setGrupo(null)} />
        <div className="racha-bar">
          <span className="racha-hint">Dices cada sonido y luego la palabra. Al lograrla se abre la siguiente</span>
        </div>
        {(() => {
          // test previo -> primeros 5 ejercicios -> prueba del medio -> el resto -> reto final
          const abrir = (def, etapa) => setPrueba({ def, etapa });
          const corte = grupo.palabras.length > EJERCICIOS_POR_EVALUACION ? EJERCICIOS_POR_EVALUACION : grupo.palabras.length;
          const boton = (p, k) => {
            const primerasOk = grupo.palabras.slice(0, corte).every((q) => logradas[q.palabra]);
            const bloqueada = grupo.key !== "pr_m" && ((k > 0 && !logradas[grupo.palabras[k - 1].palabra] && !logradas[p.palabra]) || (k >= corte && !primerasOk)); // la M queda abierta para la presentación
            return (
              <button
                key={p.palabra}
                disabled={bloqueada}
                className={`nivel-palabra ${logradas[p.palabra] ? "lograda" : ""} ${bloqueada ? "nivel-cerrado" : ""}`}
                onClick={() => setPalabra(p)}
              >
                <span className="nivel-emoji">{bloqueada ? "🔒" : p.emoji}</span>
                {p.palabra}
                {logradas[p.palabra] && <span className="nivel-check">✅</span>}
              </button>
            );
          };
          return (
            <>
              <PruebasDelSonido grupo={grupo} solo="previa" onAbrir={abrir} />
              <div className="nivel-palabras" style={{ padding: "0 14px 14px" }}>
                {grupo.palabras.slice(0, corte).map((p, k) => boton(p, k))}
              </div>
              <PruebasDelSonido grupo={grupo} solo="intermedia" onAbrir={abrir} />
              {grupo.palabras.length > corte && (
                <>
                  <div className="posicion-titulo" style={{ padding: "0 14px" }}>⭐ Palabras más difíciles</div>
                  <div className="nivel-palabras" style={{ padding: "0 14px 14px" }}>
                    {grupo.palabras.slice(corte).map((p, k) => boton(p, corte + k))}
                  </div>
                </>
              )}
              <PruebasDelSonido grupo={grupo} solo="final" onAbrir={abrir} />
            </>
          );
        })()}
      </div>
    );
  }

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🔤</span> Ejercicios</span>
      </div>
      <div className="racha-bar">
        <span className="racha-hint">Di cada sonido de la palabra y luego la palabra entera 🗣️</span>
      </div>
      <div className="etapa-list">
        {grupos.map((g, i) => (
          <button
            key={g.key}
            className={`etapa-card etapa-color-${i % 4}`}
            onClick={() => setGrupo(g)}
            style={{ animationDelay: `${i * 0.04}s` }}
          >
            <span className="etapa-card-tag">{g.palabras.length} palabras</span>
            <span className="etapa-card-word"><span className="etapa-card-icon">{g.icono}</span> {g.titulo}</span>
            <span className="etapa-card-action">Practicar →</span>
          </button>
        ))}
      </div>
    </div>
  );
}
