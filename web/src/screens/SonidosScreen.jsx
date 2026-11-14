import { useEffect, useRef, useState } from "react";
import { getServerUrl, ejercicioVoz, guardarEjercicio } from "../api.js";
import { createAudioRecorder } from "../audioRecorder.js";
import EsfuerzoResumen from "../components/EsfuerzoResumen.jsx";
import {
  EVALUACIONES, NIVELES_SILABAS, PRACTICA_SECCIONES, PANELES, ETIQUETA_POS,
  evaluacionDeGrupo, grupoDeEval, palabrasDeGrupo,
  EJERCICIOS_POR_EVALUACION, SESIONES_LEVE, SESIONES_NORMAL,
} from "../palabras.js";
import {
  contarLogradas, reiniciarSonido, diasDeUso, ejerciciosHechos, estadoEvaluacion, evaluaciones,
  guardarEvaluacion, guardarSoloImagen, marcarLograda, SONIDOS_DESBLOQUEADOS, palabrasLogradas, soloImagen,
} from "../progreso.js";
import { mensajeBien, mensajeFalta, sonidoBien, sonidoFalta } from "../sonidos.js";
import { useLandmarks } from "../components/useLandmarks.js";
import "./LearningPathScreen.css";
import "./HomeScreen.css";
import "./EjerciciosScreen.css";
import "./SonidosScreen2.css";
import FonemasScreen from "./FonemasScreen.jsx";
import { fonemasDe, sonidosAMejorar, sonidosBienDichos, videosDe, decirSonido } from "../fonemas.js";
import { GrabarSonido } from "./FonemasScreen.jsx";

// Los tests (previo, intermedio y final) están apagados por ahora: ponerlo en true para volver a mostrarlos.
const TESTS_ACTIVOS = true;
let modoRecordado = null; // "camara" | "voz": lo que se eligió la última vez
const PAUSA_RESULTADO_MS = 4500; // el resultado se queda a la vista antes de seguir

// Botón "Ver cómo se dice": al apretarlo se muestra el video de referencia (cara recortada, con
// audio) y empieza a reproducirse; se puede cerrar.
export function VideoReferencia({ src, etiqueta }) {
  const [abierto, setAbierto] = useState(false);
  const videoRef = useRef(null);
  if (!abierto) {
    return (
      <button className="ver-video" onClick={() => setAbierto(true)}>
        ▶ {etiqueta}
      </button>
    );
  }
  return (
    <div className="video-ref">
      {/* Sin barra de control: se ve seguido de principio a fin, sin adelantar ni retroceder. */}
      <video
        ref={videoRef} className="como-video" src={src} autoPlay playsInline disablePictureInPicture
        controlsList="nodownload noremoteplayback" onContextMenu={(e) => e.preventDefault()}
      />
      <button className="ver-video" onClick={() => { const v = videoRef.current; if (v) { v.currentTime = 0; v.play(); } }}>🔁 Ver otra vez</button>
      <button className="ver-video" onClick={() => setAbierto(false)}>✕ Cerrar el video</button>
    </div>
  );
}

function Encabezado({ icono, titulo, onBack }) {
  return (
    <div className="path-header">
      {onBack && <button className="icon-button" onClick={onBack}>←</button>}
      <span className="path-title">{icono} {titulo}</span>
    </div>
  );
}


// Cuando la palabra sale mal, dice EXACTAMENTE qué sonido hay que mejorar y lo marca en la palabra.
function ElogioEspecifico({ palabra, resultado, bien }) {
  const buenos = sonidosBienDichos(palabra, resultado.scores);
  if (!buenos || !buenos.length) return null;
  const nombres = buenos.map((m) => m.letras).join(" y ");
  const texto = bien
    ? `¡Muy bien ${buenos.length > 1 ? "los sonidos" : "el sonido"} ${nombres}! 🌟`
    : `${buenos.length > 1 ? "Te salieron bien" : "Te salió bien"} ${buenos.length > 1 ? "los sonidos" : "el sonido"} ${nombres} 👏`;
  return <div className="repeat-hint" style={{ marginTop: 4, fontWeight: 700, color: "#2fa56c" }}>{texto}</div>;
}

// Sonidos a mejorar de una palabra mal dicha (hasta 2). Si ninguno bajó del corte (p. ej. N/LL/Y inicial por
// el reconocimiento libre) se señala el de menor puntaje, o el primero si no hay puntajes.
function sonidosParaPracticar(palabra, resultado) {
  const letras = fonemasDe(palabra);
  let mal = sonidosAMejorar(palabra, resultado.scores);
  if (mal && !mal.length && letras.length) {
    const sc = Array.isArray(resultado.scores) && resultado.scores.length === letras.length ? resultado.scores : null;
    let i = 0;
    if (sc) sc.forEach((v, k) => { if (v != null && (sc[i] == null || v < sc[i])) i = k; });
    mal = [{ letras: letras[i].letras, indice: i, score: sc ? sc[i] : null }];
  }
  return { letras, mal: mal || [] };
}

// "Mira y repite": al fallar, el video de cómo se hace el sonido que falló y un botón para decirlo suelto.
function MiraYRepite({ palabra, resultado }) {
  const { letras, mal } = sonidosParaPracticar(palabra, resultado);
  const [paso, setPaso] = useState("ver"); // ver -> decir -> listo
  const videoRef = useRef(null);
  if (!mal.length) return null;
  const u = letras[mal[0].indice];
  const videos = videosDe(u.sonido);
  const nombre = mal[0].letras;
  return (
    <div style={{ marginTop: 10, textAlign: "center" }}>
      <div className="repeat-hint" style={{ fontWeight: 800 }}>👀 Mira cómo se hace la {nombre} y repite</div>
      {videos.length > 0 && (
        <video
          key={videos[0]} ref={videoRef} className="como-video" src={videos[0]} autoPlay playsInline disablePictureInPicture
          controlsList="nodownload noremoteplayback" onContextMenu={(e) => e.preventDefault()}
        />
      )}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap", margin: "6px 0" }}>
        <button className="ver-video" onClick={() => { const v = videoRef.current; if (v) { v.currentTime = 0; v.play(); } else decirSonido(u.sonido); }}>🔁 Ver otra vez</button>
        <button className="ver-video" onClick={() => decirSonido(u.sonido)}>🔊 Escuchar la {nombre}</button>
      </div>
      {paso === "ver" && (
        <button className="path-cta" onClick={() => setPaso("decir")}>Ahora tú →</button>
      )}
      {paso === "decir" && <GrabarSonido sonido={{ sonido: u.sonido, letras: u.letras }} onBien={() => setPaso("listo")} />}
      {paso === "listo" && <div className="repeat-hint" style={{ fontWeight: 800, color: "#2fa56c" }}>¡Muy bien la {nombre}! 🌟 Ahora intenta la palabra otra vez</div>}
    </div>
  );
}

function SonidoAPracticar({ palabra, resultado }) {
  const { letras, mal } = sonidosParaPracticar(palabra, resultado);
  if (!mal || !mal.length) return null;
  const marcados = new Set(mal.map((m) => m.indice));
  return (
    <div style={{ marginTop: 6, textAlign: "center" }}>
      <div className="repeat-hint" style={{ marginBottom: 6 }}>
        Practica este sonido: <b>{mal.map((m) => m.letras).join(" y ")}</b>
      </div>
      <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap" }}>
        {letras.map((u, k) => (
          <span
            key={k}
            style={{
              padding: "4px 11px", borderRadius: 10, fontWeight: 800, fontSize: 20,
              background: marcados.has(k) ? "#ff7a8a" : "#e4ece8", color: marcados.has(k) ? "#fff" : "#4a5560",
            }}
          >
            {u.letras}
          </span>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------ UNA PALABRA
// Muestra el dibujo (y la palabra, salvo en "solo imagen"), graba y evalúa con
// POST /ejercicio_voz. En la evaluación NO se repite el intento y, tras la pausa,
// pasa a la siguiente; en la práctica se puede intentar de nuevo.
export function PalabraStage({ item, claveGuardado, tipo, modoEvaluacion, indice, total, onTerminar, onBack, modoInicial = null, etiquetaBack = null }) {
  const streamRef = useRef(null);
  const recorderRef = useRef(createAudioRecorder());
  const timerRef = useRef(null);
  const [micListo, setMicListo] = useState(false);
  const [micError, setMicError] = useState(null);
  const [grabando, setGrabando] = useState(false);
  const [procesando, setProcesando] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [mensaje, setMensaje] = useState("");
  const [esfuerzo, setEsfuerzo] = useState(null);
  const [imagenSola] = useState(soloImagen());
  const [verPalabra, setVerPalabra] = useState(false);
  const [hayVideo, setHayVideo] = useState(false);
  const [modo, setModo] = useState(modoInicial || modoRecordado); // null = todavía no eligió
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    let cancelado = false;
    if (!modo) return undefined;
    const abrir = async () => {
      // Con cámara si está activada; si no deja la cámara, sigue solo con la voz.
      let stream = null;
      if (modo === "camara") {
        try { stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: { facingMode: "user" } }); } catch { stream = null; }
      }
      if (!stream) stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      return stream;
    };
    abrir()
      .then((stream) => {
        if (cancelado) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setHayVideo(stream.getVideoTracks().length > 0);
        setMicListo(true);
      })
      .catch((e) => !cancelado && setMicError(e.message || "No se pudo prender el micrófono."));
    return () => {
      cancelado = true;
      clearTimeout(timerRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [item.palabra, modo]);

  function empezar() {
    if (streamRef.current && recorderRef.current.start(streamRef.current)) {
      setGrabando(true);
      setResultado(null);
    }
  }

  async function terminar() {
    setGrabando(false);
    const blob = recorderRef.current.stop();
    if (!blob) return;
    const esf = recorderRef.current.esfuerzo();
    setEsfuerzo(esf);
    setProcesando(true);
    try {
      const r = await ejercicioVoz(getServerUrl(), blob, item.palabra, "palabra");
      setResultado(r);
      if (r.valid) {
        const bien = !!r.correcta;
        setMensaje(bien ? mensajeBien() : mensajeFalta());
        bien ? sonidoBien() : sonidoFalta();
        guardarEjercicio(getServerUrl(), {
          ejercicio_id: claveGuardado, tipo, correcto: bien, voz: esf?.voz,
          porcentaje_ok: r.porcentaje_ok, tipo_error: r.tipo_error,
        });
        if (!modoEvaluacion && bien) marcarLograda(item.palabra);
        if (modoEvaluacion) {
          timerRef.current = setTimeout(() => onTerminar(bien), PAUSA_RESULTADO_MS);
        }
      }
    } catch (e) {
      setResultado({ valid: false, reason: e.message || "Error al analizar el audio." });
    } finally {
      setProcesando(false);
    }
  }

  useEffect(() => {
    if (hayVideo && videoRef.current && streamRef.current) videoRef.current.srcObject = streamRef.current;
  }, [hayVideo, micListo]);
  // puntos de la cara en vivo, como en la práctica de siempre
  useLandmarks(videoRef, canvasRef, hayVideo && micListo);

  function elegir(m) {
    modoRecordado = m;
    setMicListo(false);
    setHayVideo(false);
    setResultado(null);
    setModo(m);
  }

  if (!modo) {
    return (
      <div className="path-screen">
        <Encabezado icono={item.emoji} titulo={item.palabra} onBack={onBack} />
        <div className="mode-choice-stage">
          <div className="palabra-dibujo">{item.emoji}</div>
          <div className="mode-choice-title">¿Cómo quieres practicar?</div>
          <div className="mode-choice-buttons">
            <button className="mode-choice-button" onClick={() => elegir("camara")}>
              <span className="mode-choice-icon">📷</span>
              Con cámara
            </button>
            <button className="mode-choice-button" onClick={() => elegir("voz")}>
              <span className="mode-choice-icon">🎤</span>
              Solo con la voz
            </button>
          </div>
        </div>
      </div>
    );
  }

  const bien = resultado?.valid && resultado.correcta;
  return (
    <div className="path-screen">
      <Encabezado
        icono={item.emoji}
        titulo={modoEvaluacion ? `${tipo === "diagnostico" ? "Prueba" : "Evaluación"} ${indice + 1} de ${total}` : "Practica"}
        onBack={onBack}
      />
      <div className="repeat-stage ejercicio-stage">
        <div className="palabra-dibujo">{item.emoji}</div>
        {imagenSola && !modoEvaluacion && !verPalabra ? (
          <>
            <div className="repeat-hint">¿Qué ves en el dibujo? ¡Dilo en voz alta!</div>
            <button className="cambiar-modo" onClick={() => setVerPalabra(true)}>🔤 No sé qué es: ver la palabra</button>
          </>
        ) : (
          <div className="repeat-word" style={{ fontSize: 34 }}>{item.palabra.toUpperCase()}</div>
        )}
        {modoEvaluacion && item.pos && (
          <div className="repeat-hint">Sonido: {ETIQUETA_POS[item.pos] || item.pos}</div>
        )}

        {micError ? (
          <div className="audio-only-error">No se pudo usar el micrófono: {micError}</div>
        ) : !micListo ? (
          <div className="audio-only-label">Activando el micrófono...</div>
        ) : hayVideo ? (
          <div className={`camera-frame frase-camera ${grabando ? "is-recording" : ""}`}>
            <video ref={videoRef} className="camera-preview" autoPlay muted playsInline />
            <canvas ref={canvasRef} className="camera-overlay" />
          </div>
        ) : (
          <div className="audio-only-stage" style={{ marginTop: 4 }}>
            <div className={`audio-only-orb ${grabando ? "is-recording" : ""}`}>
              {grabando && (
                <>
                  <span className="audio-only-pulse audio-only-pulse-1" />
                  <span className="audio-only-pulse audio-only-pulse-2" />
                </>
              )}
              <span className="audio-only-icon">🎤</span>
            </div>
            <span className="audio-only-label">
              {procesando ? "Analizando..." : grabando ? "Escuchando..." : "Toca para grabar"}
            </span>
          </div>
        )}

        {!grabando && !procesando && (
          <button className="cambiar-modo" onClick={() => elegir(modo === "camara" ? "voz" : "camara")}>
            {modo === "camara" ? "🎤 Cambiar a solo voz" : "📷 Cambiar a cámara"}
          </button>
        )}
        {micListo && !procesando && !(modoEvaluacion && resultado?.valid) && (
          <button className="path-cta" onClick={grabando ? terminar : empezar}>
            {grabando ? "Listo, evaluar →" : "🎤 Grabar"}
          </button>
        )}

        {resultado && (
          resultado.valid ? (
            <div className="result-overlay-tip" style={{ marginTop: 6 }}>
              <div className="result-overlay-emoji">{bien ? "😄" : "🤔"}</div>
              <span className="result-overlay-tip-label">{mensaje}</span>
              <ElogioEspecifico palabra={item.palabra} resultado={resultado} bien={bien} />
              {!bien && <SonidoAPracticar palabra={item.palabra} resultado={resultado} />}
              {!bien && !modoEvaluacion && <MiraYRepite palabra={item.palabra} resultado={resultado} />}
              <EsfuerzoResumen esfuerzo={esfuerzo} />
              {modoEvaluacion ? (
                <span className="result-overlay-tip-text">Siguiente en un momento...</span>
              ) : bien ? (
                <button className="path-cta" style={{ marginTop: 8 }} onClick={() => onTerminar(true)}>
                  Seguir →
                </button>
              ) : (
                <button className="path-cta" style={{ marginTop: 8 }} onClick={empezar}>
                  🎤 Intentar de nuevo
                </button>
              )}
            </div>
          ) : (
            <div className="audio-only-error">{resultado.reason || "No se pudo analizar."}</div>
          )
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------ EVALUACIÓN
const NOMBRE_ETAPA = { previa: "Previa", intermedia: "Intermedia", final: "Final" };

export function EvaluacionRun({ evalDef, etapa, onSalir }) {
  const [i, setI] = useState(0);
  const [bien, setBien] = useState(0);
  const [fin, setFin] = useState(false);
  const [reinicio, setReinicio] = useState(false);
  const total = evalDef.palabras.length;

  function siguiente(ok) {
    const b = bien + (ok ? 1 : 0);
    setBien(b);
    if (i + 1 < total) {
      setI(i + 1);
    } else {
      guardarEvaluacion(evalDef.key, etapa, b, total, contarLogradas(palabrasDeGrupo(grupoDeEval(evalDef.key))));
      // Menos de 2 estrellas en la prueba del medio o en el reto final: se empieza de nuevo el sonido.
      if (etapa !== "previa" && b < 2) {
        reiniciarSonido(evalDef.key, palabrasDeGrupo(grupoDeEval(evalDef.key)));
        setReinicio(true);
      }
      setFin(true);
    }
  }

  if (fin) {
    const e = evaluaciones()[evalDef.key] || {};
    return (
      <div className="path-screen">
        <Encabezado icono="📋" titulo={`Evaluación ${NOMBRE_ETAPA[etapa].toLowerCase()}: ${evalDef.titulo}`} />
        <div className="repeat-stage ejercicio-stage">
          <div className="result-overlay-emoji">{bien === total ? "🌟" : "👏"}</div>
          <div className="repeat-word" style={{ fontSize: 26 }}>
            {"⭐".repeat(bien)}{"☆".repeat(total - bien)} · {bien} de {total} palabras bien
          </div>
          {reinicio && (
            <div className="repeat-hint" style={{ margin: "8px 0" }}>
              Hoy no alcanzaron las estrellas (hacen falta 2). ¡Vamos a practicar las palabras de nuevo desde el principio! 💪
            </div>
          )}
          <Comparacion e={e} />
          <button className="path-cta" onClick={onSalir}>Listo</button>
        </div>
      </div>
    );
  }
  return (
    <PalabraStage
      key={`${etapa}-${i}`}
      item={evalDef.palabras[i]}
      claveGuardado={`${evalDef.key}_${evalDef.palabras[i].palabra}`}
      tipo="evaluacion"
      modoEvaluacion
      indice={i}
      total={total}
      onTerminar={siguiente}
      onBack={onSalir}
    />
  );
}

function Comparacion({ e }) {
  const filas = ["previa", "intermedia", "final"].filter((k) => e[k]);
  if (!filas.length) return null;
  const primero = e.previa;
  const ultimo = e.final || e.intermedia;
  return (
    <div className="comparacion">
      {filas.map((k) => (
        <div key={k} className="comparacion-fila">
          <span>{NOMBRE_ETAPA[k]}</span>
          <span className="comparacion-barra">
            <span style={{ width: `${(100 * e[k].bien) / e[k].total}%` }} />
          </span>
          <span>{e[k].bien}/{e[k].total}</span>
        </div>
      ))}
      {primero && ultimo && ultimo !== primero && (
        <div className="comparacion-nota">
          {ultimo.bien > primero.bien
            ? "¡Mejoraste desde la evaluación previa! 🎉"
            : e.final
              ? "No hubo mejora en el reto final. Te recomendamos consultar con un fonoaudiólogo 🩺"
              : ultimo.bien === primero.bien
                ? "Igual que al principio. Sigue practicando poco a poco 💪"
                : "Hoy salió más difícil. Sigue practicando 💪"}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------ PRÁCTICA POR NIVELES
function Niveles({ grupo, onBack }) {
  const [abierta, setAbierta] = useState(null);
  const [, refrescar] = useState(0);
  const logradas = palabrasLogradas();

  if (abierta) {
    return (
      <PalabraStage
        key={abierta.palabra}
        item={abierta}
        claveGuardado={`${grupo.key}_${abierta.palabra}`}
        tipo="palabra"
        onTerminar={() => { setAbierta(null); refrescar((n) => n + 1); }}
        onBack={() => setAbierta(null)}
      />
    );
  }

  // Un nivel se abre cuando se logró al menos una palabra del nivel anterior.
  const abierto = (idx) => grupo.libre || idx === 0 || grupo.niveles[idx - 1].palabras.some((p) => logradas[p.palabra]);
  return (
    <div className="path-screen">
      <Encabezado icono={grupo.icono} titulo={grupo.titulo} onBack={onBack} />
      <div className="etapa-list">
        {grupo.niveles.map((nv, idx) => {
          const ok = abierto(idx);
          return (
            <div key={nv.silabas} className={`nivel-bloque ${ok ? "" : "nivel-cerrado"}`}>
              <div className="nivel-titulo">
                {ok ? "🔓" : "🔒"} {nv.titulo || `${nv.silabas} ${nv.silabas === 1 ? "sílaba" : "sílabas"}`}
              </div>
              <div className="nivel-palabras">
                {nv.palabras.map((p) => (
                  <button
                    key={p.palabra}
                    disabled={!ok}
                    className={`nivel-palabra ${logradas[p.palabra] ? "lograda" : ""}`}
                    onClick={() => setAbierta(p)}
                  >
                    <span className="nivel-emoji">{p.emoji}</span>
                    <span>{soloImagen() ? "" : p.palabra}</span>
                    {logradas[p.palabra] && <span className="nivel-check">✅</span>}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ------------------------------------------------------------ AVISO PROFESIONAL
function AvisoProfesional() {
  const dias = diasDeUso();
  const evs = evaluaciones();
  const sinMejora = Object.values(evs).some((e) => e.final && e.previa && e.final.bien <= e.previa.bien);
  const urgente = sinMejora || dias >= SESIONES_NORMAL * 3;
  return (
    <details className={`aviso-prof ${urgente ? "aviso-urgente" : ""}`} open={urgente}>
      <summary><strong>{urgente ? "🩺 Es buen momento de ir con un fonoaudiólogo" : "🩺 Esta app es una ayuda, no un reemplazo"}</strong></summary>
      <span>
        FonoKids ayuda a practicar en casa. Normalmente hacen falta unas {SESIONES_NORMAL} sesiones para
        notar cambios ({SESIONES_LEVE} si el caso es leve). Si no mejora, consulta a un profesional; si pasan
        6 meses sin mejorar puede haber un trastorno. Cosas físicas, como el frenillo, la app no las arregla.
        A los 6–7 años la mayoría de los sonidos ya se adquieren; si no dice una palabra, hay que estimularla.
        La R, la S o la K pueden tardar más, a veces hasta los 25 años.
      </span>
      <span className="aviso-dias">Días que has practicado: {dias}</span>
    </details>
  );
}

// ------------------------------------------------------------ PANTALLA
export default function SonidosScreen({ onBack, categoria, titulo, progreso, extra, palabrasModelo = [], onAbrirModelo }) {
  const [vista, setVista] = useState(null); // { tipo: "eval", def, etapa } | { tipo: "niveles", grupo }
  const [imagenSola, setImagenSola] = useState(soloImagen());
  const [, refrescar] = useState(0);

  if (vista?.tipo === "eval") {
    return (
      <EvaluacionRun
        evalDef={vista.def}
        etapa={vista.etapa}
        onSalir={() => { setVista(null); refrescar((n) => n + 1); }}
      />
    );
  }
  if (vista?.tipo === "palabra") {
    return (
      <PalabraStage
        key={vista.clave}
        item={vista.item}
        claveGuardado={vista.clave}
        tipo="palabra"
        onTerminar={() => { setVista(null); refrescar((n) => n + 1); }}
        onBack={() => setVista(null)}
      />
    );
  }
  if (vista?.tipo === "fonemas") {
    return <FonemasScreen grupoInicial={vista.grupoKey} onSalir={() => { setVista(null); refrescar((n) => n + 1); }} />;
  }
  if (vista?.tipo === "niveles") {
    return <Niveles grupo={vista.grupo} onBack={() => { setVista(null); refrescar((n) => n + 1); }} />;
  }

  const evs = evaluaciones();
  const panel = PANELES[categoria];
  const todosGrupos = PRACTICA_SECCIONES.flatMap((s) => s.grupos);
  const grupoDe = (k) => todosGrupos.find((g) => g.key === k);

  const sinTildes = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "");
  // En qué posición del sonido está una palabra: 0 al principio, 1 en medio, 2 al final.
  function posicionDe(word, letras) {
    const w = sinTildes(word);
    if (letras.some((l) => w.startsWith(l))) return 0;
    if (letras.some((l) => w.endsWith(l))) return 2;
    return 1;
  }
  const LETRAS = { r: ["r"], s: ["s"], l: ["l"], k: ["c", "k", "qu"], g: ["g"] };

  // Los niveles (al principio / en medio / al final) de un bloque, con las palabras que ya
  // tienen modelo (cámara y voz) metidas en su posición, ordenadas de más cortas a más largas.
  function nivelesDe(b, g) {
    const niveles = g.niveles.map((nv) => ({ ...nv, palabras: nv.palabras.map((p) => ({ ...p })) }));
    const yaEsta = new Set(niveles.flatMap((nv) => nv.palabras.map((p) => sinTildes(p.palabra))));
    const sonido = b.ev === "sinfonesL" || b.ev === "sinfonesR" ? null : (b.silabas || (b.grupo === "pr_k" ? "k" : b.grupo === "pr_g" ? "g" : null));
    for (const m of palabrasModelo) {
      const w = sinTildes(m.word);
      if (m.sound === "rr" && sonido) {
        // la R fuerte (RR) va aparte, después de la R al final (si ya estaba en otra lista, se saca de ahí)
        niveles.forEach((nv) => { nv.palabras = nv.palabras.filter((p) => sinTildes(p.palabra) !== w); });
        let nivelRR = niveles.find((nv) => nv.rr);
        if (!nivelRR) { nivelRR = { titulo: "RR (la R fuerte)", palabras: [], rr: true }; niveles.push(nivelRR); }
        nivelRR.palabras.push({ palabra: m.label || m.word, emoji: m.icon, modelo: m.word });
        yaEsta.add(w);
        continue;
      }
      if (yaEsta.has(w)) {
        // si ya estaba en la lista, solo se marca que tiene cámara
        niveles.forEach((nv) => nv.palabras.forEach((p) => { if (sinTildes(p.palabra) === w) p.modelo = m.word; }));
        continue;
      }
      let idx = -1;
      if (b.sinfon) {
        idx = niveles.findIndex((nv) => w.includes(sinTildes(nv.titulo).toLowerCase()));
      } else if (sonido && b.ev !== "l") {
        if (m.sound === "rr") {
          // la R fuerte (RR) va aparte, después de la R al final
          let nivelRR = niveles.find((nv) => nv.rr);
          if (!nivelRR) { nivelRR = { titulo: "RR (la R fuerte)", palabras: [], rr: true }; niveles.push(nivelRR); }
          nivelRR.palabras.push({ palabra: m.label || m.word, emoji: m.icon, modelo: m.word });
          yaEsta.add(w);
          continue;
        }
        const pos = posicionDe(m.word, LETRAS[sonido] || [sonido]);
        const letras = LETRAS[sonido] || [sonido];
        if (letras.some((l) => w.includes(l))) idx = Math.min(pos, niveles.length - 1);
      }
      if (idx < 0) continue;
      niveles[idx].palabras.push({ palabra: m.label || m.word, emoji: m.icon, modelo: m.word });
      yaEsta.add(w);
    }
    niveles.forEach((nv) => nv.palabras.sort((a, c) => sinTildes(a.palabra).length - sinTildes(c.palabra).length));
    return niveles;
  }

  // Camino de un sonido, de lo más simple a lo más difícil:
  //   test previo -> 5 ejercicios -> prueba intermedia -> ejercicios más difíciles -> prueba final
  function camino(b) {
    const ev = TESTS_ACTIVOS ? (EVALUACIONES.find((e) => e.key === b.ev) || evaluacionDeGrupo(b.grupo)) : null;
    const g = grupoDe(b.grupo);
    const niveles = nivelesDe(b, g);
    const pasos = [];
    if (ev) pasos.push({ tipo: "eval", etapa: "previa", ev });
    const total = niveles.reduce((s, nv) => s + nv.palabras.length, 0);
    if (ev && !b.sinfon && total > EJERCICIOS_POR_EVALUACION) {
      // test previo -> primeros 5 ejercicios -> prueba intermedia -> ejercicios más difíciles -> prueba final
      let quedan = EJERCICIOS_POR_EVALUACION;
      const antes = [], despues = [];
      niveles.forEach((nv) => {
        const a = nv.palabras.slice(0, Math.max(quedan, 0));
        const d = nv.palabras.slice(a.length);
        quedan -= a.length;
        if (a.length) antes.push({ ...nv, palabras: a });
        if (d.length) despues.push({ ...nv, palabras: d, titulo: a.length ? `${nv.titulo} ⭐ más difíciles` : nv.titulo });
      });
      antes.forEach((nv, idx) => pasos.push({ tipo: "palabras", nivel: nv, grupoKey: g.key, idx }));
      pasos.push({ tipo: "eval", etapa: "intermedia", ev });
      const requiere = SONIDOS_DESBLOQUEADOS.has(ev.key) ? null : antes.flatMap((nv) => nv.palabras.map((p) => p.palabra));
      despues.forEach((nv, idx) => pasos.push({ tipo: "palabras", nivel: nv, grupoKey: g.key, idx: antes.length + idx, requiere }));
      pasos.push({ tipo: "eval", etapa: "final", ev });
      return pasos;
    }
    niveles.forEach((nv, idx) => {
      pasos.push({ tipo: "palabras", nivel: nv, grupoKey: g.key, idx });
      if (ev && !b.sinfon && idx === Math.min(1, niveles.length - 1)) pasos.push({ tipo: "eval", etapa: "intermedia", ev });
    });
    if (ev) {
      if (b.sinfon) pasos.push({ tipo: "eval", etapa: "intermedia", ev });
      pasos.push({ tipo: "eval", etapa: "final", ev });
    }
    return pasos;
  }

  function PasoEval({ paso, n }) {
    const e = evs[paso.ev.key] || {};
    const hecha = e[paso.etapa];
    const est = estadoEvaluacion(paso.ev.key, palabrasDeGrupo(grupoDeEval(paso.ev.key)), EJERCICIOS_POR_EVALUACION);
    const toca = est[paso.etapa].toca;
    const faltan = est[paso.etapa].faltan;
    const nombre = { previa: "Para empezar: ¿cómo te sale?", intermedia: "¿Cómo vas? Mira cuánto mejoraste", final: "¡Gran reto final!" }[paso.etapa];
    return (
      <button
        className={`paso-camino paso-eval ${hecha ? "paso-hecho" : ""}`}
        disabled={!toca}
        onClick={() => setVista({ tipo: "eval", def: paso.ev, etapa: paso.etapa })}
      >
        <span className="paso-num">{n}</span>
        <span className="paso-texto">
          <strong>📋 {nombre}</strong>
          <small>
            {hecha ? `${"⭐".repeat(hecha.bien)}${"☆".repeat(Math.max(hecha.total - hecha.bien, 0))}  ${hecha.bien} de ${hecha.total}`
              : toca ? "Toca hacerla ahora →"
              : paso.etapa === "intermedia" ? `Se abre al lograr ${faltan} palabra${faltan === 1 ? "" : "s"} más`
              : `Se abre al lograr todas las palabras (faltan ${faltan})`}
          </small>
        </span>
      </button>
    );
  }

  function PasoPalabras({ paso }) {
    const logradas = palabrasLogradas();
    const bloqueado = !!paso.requiere && !paso.requiere.every((w) => logradas[w]);
    return (
      <div className="posicion-bloque">
        <div className="posicion-titulo">{paso.nivel.titulo}</div>
        {paso.nivel.rr && (
          <>
            <VideoReferencia src="/referencia/rr.mp4" etiqueta="Ver cómo se dice la RR (video 1)" />
            <VideoReferencia src="/referencia/rrb1.mp4" etiqueta="Ver cómo se dice la RR (video 2)" />
            <VideoReferencia src="/referencia/rrb2.mp4" etiqueta="Ver cómo se dice la RR (video 3)" />
          </>
        )}
        {bloqueado && <div className="eval-nota" style={{ textAlign: "left" }}>🔒 Se abren al lograr las primeras {paso.requiere.length} palabras</div>}
        <div className="nivel-palabras">
          {paso.nivel.palabras.map((p) => (
            <button
              key={p.palabra}
              disabled={bloqueado}
              className={`nivel-palabra ${logradas[p.palabra] ? "lograda" : ""} ${bloqueado ? "nivel-cerrado" : ""}`}
              onClick={() => (p.modelo && onAbrirModelo
                ? onAbrirModelo(p.modelo)
                : setVista({ tipo: "palabra", item: p, clave: `${paso.grupoKey}_${p.palabra}` }))}
            >
              <span className="nivel-emoji">{p.emoji}</span>
              <span>{soloImagen() ? "" : p.palabra}</span>
              {p.modelo && <span className="nivel-camara" title="Con cámara y voz">🎥</span>}
              {logradas[p.palabra] && <span className="nivel-check">✅</span>}
            </button>
          ))}
        </div>
      </div>
    );
  }

  const tarjetaGrupo = (g, i) => (
    <button
      key={g.key}
      className={`etapa-card etapa-color-${(i + 1) % 4}`}
      style={{ width: "100%", marginBottom: 8 }}
      onClick={() => setVista({ tipo: "niveles", grupo: g })}
    >
      <span className="etapa-card-word"><span className="etapa-card-icon">{g.icono}</span> {g.titulo}</span>
      <span className="etapa-card-action">Practicar →</span>
    </button>
  );

  return (
    <div className="path-screen">
      <div className="path-header">
        {onBack && <button className="icon-button" onClick={onBack}>←</button>}
        <span className="path-title">{titulo || (panel ? `Test y práctica: ${panel.titulo}` : "Test y práctica")}</span>
        {progreso && <span className="path-progress-pill">{progreso}</span>}
      </div>
      <div className="etapa-list" style={{ paddingBottom: 24 }}>
        <AvisoProfesional />
        <label className="solo-imagen">
          <input
            type="checkbox"
            checked={imagenSola}
            onChange={(e) => { guardarSoloImagen(e.target.checked); setImagenSola(e.target.checked); }}
          />
          Para niños que no leen: mostrar solo el dibujo
        </label>

        {panel && panel.bloques.map((b) => (
          <div key={b.grupo} className="camino-bloque">
            <div className="seccion-titulo">{b.nombre}</div>
            {b.grupo.startsWith("pr_") && (
              <button className="ver-video" onClick={() => setVista({ tipo: "fonemas", grupoKey: b.grupo })}>
                🔤 Ejercicio: di cada sonido de las palabras
              </button>
            )}
            {b.grupo === "pr_r" && <VideoReferencia src="/referencia/r.mp4" etiqueta="Ver cómo se dice la R" />}
            {b.grupo === "pr_k" && <VideoReferencia src="/referencia/c1.mp4" etiqueta="Ver cómo se dice la C" />}
            {b.grupo === "pr_s" && (
              <>
                <VideoReferencia src="/referencia/s1.mp4" etiqueta="Ver cómo se dice la S (video 1)" />
                <VideoReferencia src="/referencia/s2.mp4" etiqueta="Ver cómo se dice la S (video 2)" />
              </>
            )}
            {b.grupo === "pr_l" && (
              <>
                <VideoReferencia src="/referencia/l1.mp4" etiqueta="Ver cómo se dice la L (video 1)" />
                <VideoReferencia src="/referencia/l2.mp4" etiqueta="Ver cómo se dice la L (video 2)" />
              </>
            )}
            {b.grupo === "pr_ll" && (
              <>
                <VideoReferencia src="/referencia/ll1.mp4" etiqueta="Ver cómo se dice la LL (video 1)" />
                <VideoReferencia src="/referencia/ll2.mp4" etiqueta="Ver cómo se dice la LL (video 2)" />
              </>
            )}
            {b.grupo === "pr_ch" && (
              <>
                <VideoReferencia src="/referencia/ch1.mp4" etiqueta="Ver cómo se dice la CH (video 1)" />
                <VideoReferencia src="/referencia/ch2.mp4" etiqueta="Ver cómo se dice la CH (video 2)" />
              </>
            )}
            {b.grupo === "pr_d" && (
              <>
                <VideoReferencia src="/referencia/d1.mp4" etiqueta="Ver cómo se dice la D (video 1)" />
                <VideoReferencia src="/referencia/d2.mp4" etiqueta="Ver cómo se dice la D (video 2)" />
              </>
            )}
            {b.grupo === "pr_n" && (
              <>
                <VideoReferencia src="/referencia/n1.mp4" etiqueta="Ver cómo se dice la N (video 1)" />
                <VideoReferencia src="/referencia/n2.mp4" etiqueta="Ver cómo se dice la N (video 2)" />
              </>
            )}
            {b.grupo === "pr_p" && (
              <>
                <VideoReferencia src="/referencia/p1.mp4" etiqueta="Ver cómo se dice la P (video 1)" />
                <VideoReferencia src="/referencia/p2.mp4" etiqueta="Ver cómo se dice la P (video 2)" />
              </>
            )}
            {b.grupo === "pr_z" && (
              <>
                <VideoReferencia src="/referencia/z1.mp4" etiqueta="Ver cómo se dice la Z (video 1)" />
                <VideoReferencia src="/referencia/z2.mp4" etiqueta="Ver cómo se dice la Z (video 2)" />
              </>
            )}
            {b.grupo === "pr_y" && (
              <>
                <VideoReferencia src="/referencia/y1.mp4" etiqueta="Ver cómo se dice la Y (video 1)" />
                <VideoReferencia src="/referencia/y2.mp4" etiqueta="Ver cómo se dice la Y (video 2)" />
              </>
            )}
            {b.grupo === "pr_ni" && (
              <>
                <VideoReferencia src="/referencia/ni1.mp4" etiqueta="Ver cómo se dice la Ñ (video 1)" />
                <VideoReferencia src="/referencia/ni2.mp4" etiqueta="Ver cómo se dice la Ñ (video 2)" />
              </>
            )}
            {b.grupo === "pr_t" && (
              <>
                <VideoReferencia src="/referencia/t1.mp4" etiqueta="Ver cómo se dice la T (video 1)" />
                <VideoReferencia src="/referencia/t2.mp4" etiqueta="Ver cómo se dice la T (video 2)" />
              </>
            )}
            {b.grupo === "pr_b" && (
              <>
                <VideoReferencia src="/referencia/b1.mp4" etiqueta="Ver cómo se dice la B (video 1)" />
                <VideoReferencia src="/referencia/b2.mp4" etiqueta="Ver cómo se dice la B (video 2)" />
              </>
            )}
            {b.grupo === "pr_j" && (
              <>
                <VideoReferencia src="/referencia/j1.mp4" etiqueta="Ver cómo se dice la J (video 1)" />
                <VideoReferencia src="/referencia/j2.mp4" etiqueta="Ver cómo se dice la J (video 2)" />
              </>
            )}
            {b.grupo === "pr_m" && (
              <>
                <VideoReferencia src="/referencia/m1.mp4" etiqueta="Ver cómo se dice la M (video 1)" />
                <VideoReferencia src="/referencia/m2.mp4" etiqueta="Ver cómo se dice la M (video 2)" />
              </>
            )}
            {b.grupo === "pr_f" && (
              <>
                <VideoReferencia src="/referencia/f1.mp4" etiqueta="Ver cómo se dice la F (video 1)" />
                <VideoReferencia src="/referencia/f2.mp4" etiqueta="Ver cómo se dice la F (video 2)" />
              </>
            )}
            {b.grupo === "pr_g" && (
              <>
                <VideoReferencia src="/referencia/g1.mp4" etiqueta="Ver cómo se dice la G (video 1)" />
                <VideoReferencia src="/referencia/g2.mp4" etiqueta="Ver cómo se dice la G (video 2)" />
              </>
            )}
            {camino(b).map((paso, i) =>
              paso.tipo === "eval"
                ? <PasoEval key={`${paso.ev.key}-${paso.etapa}`} paso={paso} n={{ previa: 1, intermedia: 2, final: 3 }[paso.etapa]} />
                : <PasoPalabras key={paso.nivel.titulo} paso={paso} />
            )}
          </div>
        ))}

        {extra}

        {panel && panel.bloques.some((b) => b.directas) && (
          <>
            <div className="seccion-titulo">Para reforzar: sílabas directas e inversas</div>
            {panel.bloques.filter((b) => b.directas).map((b, i) => tarjetaGrupo(grupoDe(b.directas), i))}
          </>
        )}
        {panel && panel.bloques.some((b) => b.silabas) && (
          <>
            <div className="seccion-titulo">Para reforzar: palabras de 1 a 4 sílabas</div>
            {panel.bloques.filter((b) => b.silabas).map((b, i) => tarjetaGrupo(NIVELES_SILABAS.find((n) => n.key === b.silabas), i))}
          </>
        )}

        <div className="eval-nota" style={{ textAlign: "center" }}>
          Ejercicios hechos hasta ahora: {ejerciciosHechos()}
        </div>
      </div>
    </div>
  );
}
