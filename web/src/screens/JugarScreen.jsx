import { useEffect, useState } from "react";
import Mascota, { BotonVoz } from "../components/Mascota.jsx";
import { PalabraStage, EvaluacionRun } from "./SonidosScreen.jsx";
import EjerciciosScreen from "./EjerciciosScreen.jsx";
import { PalabraPorSonidos } from "./FonemasScreen.jsx";
import { armarPartida, palabrasAprendidas, guardarTestPrevio, testPrevioHecho, gruposDelCamino, getNombre, setNombre, entrarNino } from "../modoNinos.js";
import { decirSonido, pistaCara } from "../fonemas.js";
import { callar, hablar, hablarUnaVez } from "../voz.js";
import { getEstado } from "../rachas.js";
import { useAyuda } from "../useAyuda.js";
import { usarCamara } from "../progreso.js";
import "./LearningPathScreen.css";
import "./JugarScreen.css";

// Modo niños: todo guiado. Fofo elige la letra y las palabras, habla en voz alta y el niño solo toca
// botones grandes con dibujos. Lo de los papás (avance, tablas) queda detrás de una pregunta de sumar.

// Cómo se dice el nombre de cada sonido del juego (para hablarlo y para el cuadro del resumen).
const nombreHablado = (p) => ({ sf_l: "las letras seguidas con ele", sf_r: "las letras seguidas con erre" }[p.grupo] || p.titulo.replace(/\s*\/\s*/g, " o "));
const etiquetaCorta = (p) => ({ sf_l: "PL BL…", sf_r: "PR BR…" }[p.grupo] || p.titulo);

// El test previo: una palabra por cada sonido difícil. No se repite ni se corrige: solo se anota qué sonidos necesitan práctica.
function TestPrevio({ palabras, onListo, onSalir }) {
  const [i, setI] = useState(0);
  const [res, setRes] = useState([]);
  const [fin, setFin] = useState(false);
  const bienPorGrupo = {};
  for (const r of res) bienPorGrupo[r.grupo] = (bienPorGrupo[r.grupo] ?? true) && r.bien;

  function siguiente(bien) {
    const nuevo = [...res, { grupo: palabras[i].grupo, bien }];
    setRes(nuevo);
    if (i + 1 < palabras.length) { setI(i + 1); return; }
    const porGrupo = {};
    for (const r of nuevo) porGrupo[r.grupo] = (porGrupo[r.grupo] ?? true) && r.bien;
    guardarTestPrevio(porGrupo);
    setFin(true);
    const mal = Object.values(porGrupo).filter((v) => v === false).length;
    const nombresMal = palabras.filter((p, k, arr) => porGrupo[p.grupo] === false && arr.findIndex((x) => x.grupo === p.grupo) === k).map((p) => nombreHablado(p));
    const cuantosBien = Object.values(porGrupo).filter((v) => v === true).length;
    hablar(mal
      ? `¡Listo! Dijiste bien ${cuantosBien} sonidos. Vamos a practicar: ${nombresMal.join(", ")}.`
      : "¡Wow! ¡Lo dijiste todo muy bien!", { fuerte: !mal });
  }

  if (fin) {
    // Un cuadro por cada sonido del juego, con su dibujo: los que hay que practicar primero y los que salieron bien después.
    const sonidos = palabras.filter((p, k, arr) => arr.findIndex((x) => x.grupo === p.grupo) === k);
    const aPracticar = sonidos.filter((p) => bienPorGrupo[p.grupo] === false);
    const biens = sonidos.filter((p) => bienPorGrupo[p.grupo] !== false);
    const Cuadro = ({ p, ok }) => (
      <div className={`resumen-cuadro ${ok ? "resumen-ok" : "resumen-mal"}`} title={p.palabra}>
        <span className="resumen-dibujo">{p.emoji}</span>
        <b>{etiquetaCorta(p)}</b>
        <span className="resumen-marca">{ok ? "⭐" : "💪"}</span>
      </div>
    );
    return (
      <div className="path-screen">
        <div className="jugar-centro resumen-centro">
          <Mascota animo="festejo" mensaje={aPracticar.length ? `¡Listo! Dijiste bien ${biens.length} y vamos a practicar ${aPracticar.length}.` : "¡Wow! ¡Lo dijiste todo muy bien!"} />
          {aPracticar.length > 0 && (
            <section className="resumen-seccion">
              <h3 className="resumen-titulo resumen-titulo-mal">💪 A practicar</h3>
              <div className="resumen-rejilla">{aPracticar.map((p) => <Cuadro key={p.grupo} p={p} ok={false} />)}</div>
            </section>
          )}
          {biens.length > 0 && (
            <section className="resumen-seccion">
              <h3 className="resumen-titulo resumen-titulo-ok">⭐ ¡Lo dijiste bien!</h3>
              <div className="resumen-rejilla">{biens.map((p) => <Cuadro key={p.grupo} p={p} ok />)}</div>
            </section>
          )}
          <button className="jugar-btn guia-mano" onClick={onListo}>🗺️ ¡Ver mi camino!</button>
        </div>
      </div>
    );
  }
  const item = palabras[i];
  return (
    <>
      <div className="test-barra" aria-label={`Palabra ${i + 1} de ${palabras.length}`}><span style={{ width: `${(100 * i) / palabras.length}%` }} /></div>
      <PalabraStage
        key={`${item.grupo}-${item.palabra}`}
        item={item}
        claveGuardado={`testprevio_${item.palabra}`}
        tipo="diagnostico"
        modoEvaluacion
        modoNinos
        modoInicial={usarCamara() ? "camara" : "voz"}
        indice={i}
        total={palabras.length}
        onTerminar={siguiente}
        onBack={onSalir}
      />
    </>
  );
}

const L = (p) => p.titulo.replace(/\s*\/\s*/g, " o ");
// Lo que Fofo dice antes de cada partida: prueba previa, del medio, final o práctica de la letra.
function textoIntro(p) {
  if (p.tipo === "testprevio") return { voz: `¡Primero vamos a jugar con ${p.palabras.length} palabras! Yo te escucho.`, texto: `¡Primero vamos a jugar con ${p.palabras.length} palabras!`, rotulo: "Juego de Fofo" };
  if (p.tipo === "prueba") {
    if (p.etapa === "previa") return { voz: `Primero vamos a ver cómo te sale la letra ${L(p)}. ¡Son tres palabritas!`, texto: `¡Veamos cómo te sale la ${p.titulo}! 3 palabritas`, rotulo: "Juego de Fofo" };
    if (p.etapa === "intermedia") return { voz: `¡Vamos a ver cuánto mejoraste con la letra ${L(p)}!`, texto: `¡Mira cuánto mejoraste con la ${p.titulo}!`, rotulo: "¿Cuánto mejoraste?" };
    return { voz: `¡Llegó el reto final de la letra ${L(p)}! ¡Tú puedes!`, texto: `¡Reto final de la ${p.titulo}!`, rotulo: "Reto final" };
  }
  return { voz: `Vamos a jugar con la letra ${L(p)}.`, texto: `¡Vamos a jugar con la letra ${p.titulo}!`, rotulo: `La letra ${p.titulo}` };
}

const PASOS = [
  { texto: "Mira el dibujo", voz: "Paso uno. Mira el dibujo. Este es un mono." },
  { texto: "Toca el micrófono y di la palabra", voz: "Paso dos. Toca el micrófono y di la palabra en voz alta: mono." },
  { texto: "¡Si lo dices bien, ganas una estrella!", voz: "Paso tres. Si lo dices bien, ¡ganas una estrella!" },
  { texto: "Si no sale, Fofo te ayuda", voz: "Y si no sale, no te preocupes. Fofo te ayuda a decirlo con calma." },
];

// Cada paso del tutorial es una escena que muestra lo que va a pasar de verdad en el juego.
function Escena({ paso }) {
  if (paso === 0) {
    return (
      <div className="esc esc-dibujo">
        <span className="esc-ojos">👀</span>
        <div className="esc-tarjeta"><span className="esc-emoji">🐒</span><span className="esc-palabra">MONO</span></div>
      </div>
    );
  }
  if (paso === 1) {
    return (
      <div className="esc esc-mic">
        <div className="esc-dice">¡Mono!</div>
        <div className="esc-boton"><span>🎤</span><i className="onda o1" /><i className="onda o2" /></div>
        <span className="esc-dedo">👆</span>
      </div>
    );
  }
  if (paso === 2) {
    return (
      <div className="esc esc-estrella">
        <span className="esc-estrella-grande">⭐</span>
        <div className="esc-chispas"><i>✨</i><i>✨</i><i>✨</i><i>✨</i></div>
        <div className="esc-fila">⭐⭐⭐⭐⭐</div>
      </div>
    );
  }
  return (
    <div className="esc esc-ayuda">
      <div className="esc-calma">🐢 Con calma</div>
      <div className="esc-letras"><b>M</b><b>O</b><b>N</b><b>O</b></div>
      <div className="esc-boca">👄 Los labios se juntan fuerte: mmm</div>
    </div>
  );
}

// "¿Cómo se juega?": cuatro escenas con dibujo grande y voz, una tras otra. Se ve la primera vez y desde el botón ❓.
function Tutorial({ onListo }) {
  const [paso, setPaso] = useState(0);
  useEffect(() => {
    hablarUnaVez(`tuto-${paso}`, PASOS[paso].voz);
    if (paso >= PASOS.length - 1) return undefined;
    const t = setTimeout(() => setPaso(paso + 1), 5200);
    return () => clearTimeout(t);
  }, [paso]);
  const p = PASOS[paso];
  return (
    <div className="path-screen">
      <div className="jugar-centro">
        <Mascota grande mensaje="¡Así se juega!" />
        <div className="tuto-paso" key={paso}>
          <span className="tuto-num">{paso + 1}</span>
          <Escena paso={paso} />
          <span className="tuto-texto">{p.texto}</span>
        </div>
        <div className="tuto-puntos">{PASOS.map((_, k) => <i key={k} className={k === paso ? "on" : ""} />)}</div>
        <button className={`jugar-btn ${paso >= PASOS.length - 1 ? "guia-mano" : "jugar-btn-sec"}`} onClick={() => { callar(); onListo(); }}>
          {paso >= PASOS.length - 1 ? "▶ ¡A jugar!" : "Saltar ➜"}
        </button>
      </div>
    </div>
  );
}

// El camino: un círculo por cada sonido del test previo. El que brilla es el que Fofo recomienda hoy.
function Camino({ nombre, onElegir, onVolver }) {
  const grupos = gruposDelCamino();
  const sig = armarPartida();
  const claveSig = sig && sig.tipo !== "testprevio" ? sig.clave : null;
  const sugerido = grupos.find((g) => g.clave === claveSig);
  const mensaje = sugerido
    ? `¡${nombre ? nombre + ", " : ""}toca el círculo que brilla ${sugerido.icono}!`
    : "¡Wow! ¡Ya sabes todo! Toca el círculo que quieras para jugar.";
  return (
    <div className="path-screen">
      <div className="path-header">
        <button className="icon-button" onClick={onVolver}>←</button>
        <span className="path-title">🗺️ Mi camino</span>
        <BotonVoz />
      </div>
      <div className="mapa-saludo"><Mascota grande mensaje={mensaje} /></div>
      <div className="mapa-aventura">
        {grupos.map((g, i) => (
          <div
            key={g.clave}
            className={`nodo-wrap nodo-pos-${i % 4} ${g.clave === claveSig ? "nodo-sugerido" : ""}`}
            style={{ animationDelay: `${i * 0.04}s` }}
            ref={g.clave === claveSig ? (el) => { if (el) setTimeout(() => el.scrollIntoView({ block: "center", behavior: "smooth" }), 600); } : undefined}
          >
            <button
              className={`nodo nodo-color-${i % 4} ${g.estado === true ? "nodo-ok" : "nodo-practica"}`}
              onClick={() => { hablar(g.estado === true ? "¡Este ya lo sabes! Pero practiquemos otra vez." : "¡Vamos a practicar!"); onElegir(g.clave); }}
              aria-label={g.titulo}
            >
              <span className="nodo-num">{i + 1}</span>
              <span className="nodo-icono">{g.icono}</span>
              <span className="nodo-estado">{g.estado === true ? "⭐" : "💪"}</span>
            </button>
            <span className="nodo-titulo">{g.titulo}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// La pantalla de bienvenida se muestra una sola vez por visita (el primer toque además deja que el navegador hable).
let yaSaludo = false;
const TUTO_KEY = "fonokids_tuto_visto";
const vioTutorial = () => { try { return localStorage.getItem(TUTO_KEY) === "1"; } catch { return false; } };
const marcarTutorial = () => { try { localStorage.setItem(TUTO_KEY, "1"); } catch { /* nada */ } };

export default function JugarScreen({ onAprender = null, claveInicial = null, onSalir = null }) {
  const [vista, setVista] = useState(claveInicial ? "arrancando" : (yaSaludo ? "inicio" : "bienvenida")); // inicio | todo | intro | jugando | fin | estrellas | gimnasia
  const [partida, setPartida] = useState(null);
  const [i, setI] = useState(0);
  const [aciertos, setAciertos] = useState(0);
  const [ayuda, setAyuda] = useState(null); // palabra con la que se está practicando "con calma" (letra por letra)
  const [nombre, setNombreEstado] = useState(getNombre());
  const estado = getEstado();

  useEffect(() => () => callar(), []);
  // Pantalla de juego (se llega desde el camino): arranca directo con el sonido elegido.
  useEffect(() => { if (claveInicial) { const [c, m] = claveInicial.split("|"); jugar(c, m || null); } /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  // Volver: desde el camino de Aprender vuelve a Aprender; en la entrada vuelve al inicio.
  const salir = () => { callar(); if (onSalir) onSalir(); else setVista(testPrevioHecho() ? "inicio" : "bienvenida"); };
  // Qué sigue después de saludar / poner el nombre / ver el tutorial.
  function seguirDespues(deTutorial = false) {
    if (!deTutorial && !vioTutorial()) { setVista("tutorial"); return; }
    if (!testPrevioHecho()) { jugar(); return; }
    if (onAprender) onAprender(); else setVista("inicio");
  }
  useAyuda("¡Toca el botón naranja que dice inicio!", vista === "bienvenida");
  useAyuda("Escribe tu nombre aquí y toca el botón verde.", vista === "nombre");
  useAyuda("Toca el botón verde para jugar otra vez.", vista === "fin");
  useAyuda("Toca el botón verde para ver tu camino.", vista === "inicio");
  useAyuda("Toca el botón verde para empezar.", vista === "intro");

  function jugar(clave = null, modo = null) {
    const p = armarPartida(clave, modo);
    if (!p) { if (onSalir) onSalir(); else setVista("todo"); return; }
    setPartida(p); setI(0); setAciertos(0); setVista("intro");
    hablarUnaVez(`intro-${p.clave}-${p.tipo}`, `${textoIntro(p).voz} Mira y escucha cómo se hace.`);
  }

  function terminarPalabra(bien) {
    const nuevos = aciertos + (bien ? 1 : 0);
    setAciertos(nuevos);
    if (i + 1 >= partida.palabras.length) {
      setVista("fin");
      hablar(nuevos >= 3 ? "¡Lo lograste! ¡Eres un campeón!" : "¡Muy bien! ¡Sigue practicando y lo lograrás!", { fuerte: nuevos >= 3 });
    } else {
      setI(i + 1);
    }
  }

  if (vista === "bienvenida") {
    return (
      <div className="path-screen">
        <div className="jugar-centro">
          <Mascota grande animo="festejo" mensaje="¡Hola! Soy Fofo. ¡Toca aquí para empezar!" />
          <button
            className="jugar-btn jugar-btn-gigante guia-mano"
            onClick={() => {
              yaSaludo = true;
              if (!getNombre()) { setVista("nombre"); hablar("¡Hola! Soy Fofo. ¿Cómo te llamas? Escribe tu nombre."); }
              else seguirDespues();
            }}
          >
            ▶ ¡Inicio!
          </button>
        </div>
      </div>
    );
  }

  if (vista === "nombre") {
    return (
      <div className="path-screen">
        <div className="jugar-centro">
          <Mascota grande mensaje="¡Cuéntame! ¿Cómo te llamas? Escribe tu nombre." />
          <div className="nombre-rotulo">✏️ ¡Escribe tu nombre aquí! 👇</div>
          <div className={`nombre-caja ${nombre.trim() ? "" : "guia-mano"}`}>
            <input
              className="nombre-input" value={nombre} maxLength={14} autoFocus placeholder="Mi nombre es…"
              onChange={(e) => setNombreEstado(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") document.getElementById("nombre-ok")?.click(); }}
            />
          </div>
          <button
            id="nombre-ok" className={`jugar-btn ${nombre.trim() ? "guia-mano" : "boton-apagado"}`}
            disabled={!nombre.trim()}
            onClick={async () => {
              if (!nombre.trim()) return;
              // Si ya había jugado con este nombre, se recupera su avance y no repite el juego de entrada.
              const { nuevo } = await entrarNino(nombre);
              setNombre(nombre);
              hablar(nuevo ? `¡Mucho gusto, ${nombre.trim()}!` : `¡Qué bueno verte otra vez, ${nombre.trim()}!`);
              seguirDespues();
            }}
          >
            ▶ ¡Listo!
          </button>
        </div>
      </div>
    );
  }

  if (vista === "tutorial") {
    return <Tutorial onListo={() => { marcarTutorial(); seguirDespues(true); }} />;
  }

  if (vista === "camino") {
    return <Camino nombre={nombre} onElegir={(clave) => jugar(clave)} onVolver={salir} />;
  }

  if (vista === "gimnasia") {
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setVista("inicio")}>←</button>
          <span className="path-title">🤸 Gimnasia de la boca</span>
        </div>
        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}><EjerciciosScreen /></div>
      </div>
    );
  }

  if (vista === "estrellas") {
    const lista = palabrasAprendidas();
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setVista("inicio")}>←</button>
          <span className="path-title">⭐ Mis estrellas</span>
        </div>
        <div className="jugar-centro">
          <Mascota grande animo="festejo" mensaje={lista.length ? `¡Mira! Ya sabes decir ${lista.length} palabras.` : "¡Juega y gana tu primera estrella!"} />
          <div className="jugar-estrellas-grande">⭐ {estado.estrellas}</div>
          <div className="jugar-aprendidas">
            {lista.map((p) => <span key={p.palabra} className="jugar-aprendida" title={p.palabra}>{p.emoji}</span>)}
          </div>
        </div>
      </div>
    );
  }

  if (vista === "intro" && partida) {
    const pista = pistaCara(partida.sonido);
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={salir}>←</button>
          <span className="path-title">{partida.icono} {textoIntro(partida).rotulo}</span>
          <BotonVoz />
        </div>
        <div className="jugar-centro">
          <div className="jugar-letra">{partida.icono}</div>
          <Mascota grande mensaje={textoIntro(partida).texto} />
          {pista && <div className="pista-cara jugar-pista"><span className="pista-parte">{pista.parte}</span> {pista.texto}</div>}
          {partida.sonido && (
            <button className="jugar-btn jugar-btn-sec" onClick={() => decirSonido(partida.sonido)}>🔊 Escucha</button>
          )}
          <button className="jugar-btn guia-mano" onClick={() => { callar(); setVista("jugando"); }}>▶ ¡Empezar!</button>
        </div>
      </div>
    );
  }

  if (vista === "jugando" && partida && partida.tipo === "prueba") {
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={salir}>←</button>
          <span className="path-title">🎮 {textoIntro(partida).rotulo}</span>
          <BotonVoz />
        </div>
        <div className="jugar-juego">
          <EvaluacionRun
            key={`${partida.clave}-${partida.etapa}`}
            evalDef={partida.evalDef}
            etapa={partida.etapa}
            modoNinos
            modoInicial={usarCamara() ? "camara" : "voz"}
            onSalir={() => { callar(); jugar(partida.clave); }}
          />
        </div>
      </div>
    );
  }

  if (vista === "jugando" && partida && partida.tipo === "testprevio") {
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={salir}>←</button>
          <span className="path-title">🎮 Juego de Fofo</span>
          <BotonVoz />
        </div>
        <div className="jugar-juego">
          <TestPrevio palabras={partida.palabras} onListo={() => { callar(); if (onAprender) onAprender(); else setVista("camino"); }} onSalir={salir} />
        </div>
      </div>
    );
  }

  if (vista === "jugando" && partida && ayuda) {
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => { callar(); setAyuda(null); }}>←</button>
          <span className="path-title">🐢 Con calma</span>
          <BotonVoz />
        </div>
        <div className="jugar-juego">
          <PalabraPorSonidos
            key={`ayuda-${ayuda.palabra}`}
            item={ayuda}
            clave={`jugar_calma_${ayuda.palabra}`}
            modoNinos
            modoInicial={usarCamara() ? "camara" : "voz"}
            onBack={() => { callar(); setAyuda(null); }}
            onTerminar={(bien) => { callar(); setAyuda(null); terminarPalabra(bien); }}
          />
        </div>
      </div>
    );
  }

  if (vista === "jugando" && partida) {
    const item = partida.palabras[i];
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={salir}>←</button>
          <span className="jugar-progreso" aria-label={`Palabra ${i + 1} de ${partida.palabras.length}`}>
            {partida.palabras.map((_, k) => (
              <span key={k} className={k === i ? "est-ahora" : k > i ? "est-off" : ""}>{k < i ? "⭐" : "☆"}</span>
            ))}
          </span>
          <BotonVoz />
        </div>
        <div className="jugar-juego">
        <PalabraStage
          key={`${partida.clave}-${i}-${item.palabra}`}
          item={item}
          claveGuardado={`jugar_${partida.clave}_${item.palabra}`}
          tipo="practica"
          modoNinos
          modoInicial={usarCamara() ? "camara" : "voz"}
          indice={i}
          total={partida.palabras.length}
          onTerminar={terminarPalabra}
          onSaltar={() => terminarPalabra(false)}
          onAyuda={() => setAyuda(item)}
          onBack={salir}
        />
        </div>
      </div>
    );
  }

  if (vista === "fin") {
    return (
      <div className="path-screen">
        <div className="jugar-centro">
          <Mascota grande animo="festejo" mensaje={aciertos >= 3 ? "¡Lo lograste! ¡Eres un campeón!" : "¡Muy bien! ¡Sigue practicando!"} />
          <div className="jugar-estrellas-grande">{"⭐".repeat(aciertos) || "💪"}</div>
          <button className="jugar-btn guia-mano" onClick={() => jugar(partida?.clave)}>🔁 Jugar otra vez</button>
          <button className="jugar-btn jugar-btn-sec" onClick={salir}>🗺️ Mi camino</button>
        </div>
      </div>
    );
  }

  if (vista === "arrancando") return <div className="path-screen" />;

  // inicio (o "todo": ya aprendió todas las palabras)
  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🦊</span> FonoKids</span>
        <span className="jugar-esquina"><BotonVoz /></span>
      </div>
      <div className="jugar-centro">
        <Mascota grande mensaje={vista === "todo" ? "¡Wow! ¡Ya sabes todas las palabras! Eres increíble." : `¡Hola${nombre ? `, ${nombre}` : ""}! Soy Fofo. ¿Jugamos a decir palabras?`} />
        <button className="jugar-btn jugar-btn-gigante guia-mano" onClick={() => (testPrevioHecho() ? setVista("camino") : jugar())}>▶ ¡Jugar!</button>
        <div className="jugar-fila">
          <button className="jugar-btn jugar-btn-sec" onClick={() => setVista("gimnasia")}><span>🤸</span>Gimnasia</button>
          <button className="jugar-btn jugar-btn-sec" onClick={() => setVista("estrellas")}><span>⭐</span>Mis estrellas</button>
        </div>
        <button className="jugar-ayuda" onClick={() => setVista("tutorial")}>❓ ¿Cómo se juega?</button>
      </div>
    </div>
  );
}
