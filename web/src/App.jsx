import { useEffect, useState } from "react";
import LearningPathScreen from "./screens/LearningPathScreen.jsx";
import EjerciciosScreen from "./screens/EjerciciosScreen.jsx";
import LetrasScreen from "./screens/LetrasScreen.jsx";
import JugarScreen from "./screens/JugarScreen.jsx";
import { testPrevioHecho, getNombre, setNombre, reiniciarTestPrevio } from "./modoNinos.js";

// Para las pruebas y la muestra: cada vez que se actualiza la página se pide de nuevo el nombre del niño y se hace el test previo.
// Poner en false para que la app recuerde el nombre y el test entre una visita y otra.
const PEDIR_NOMBRE_Y_TEST_AL_RECARGAR = true;
if (PEDIR_NOMBRE_Y_TEST_AL_RECARGAR) {
  setNombre("");
  reiniciarTestPrevio();
  if (window.location.hash && window.location.hash !== "#/") window.history.replaceState(null, "", "#/");
}
import "./screens/NinosTema.css";
import "./App.css";

// Recorrido del niño:
//   1. Al abrir la app: "¡Hola!" -> su nombre -> cómo se juega -> TEST PREVIO (todo guiado por Fofo).
//   2. Al terminar el test cae en APRENDER (el camino de círculos), donde brilla el sonido que toca practicar.
//   3. Al tocar un círculo, Fofo guía la práctica de ese sonido (palabras, prueba del medio y reto final).
// Las otras pestañas del menú: Gimnasia de la boca.
const TABS = [
  { key: "aprender", label: "Aprender", icon: "🧭" },
  { key: "letras", label: "Cómo se dice", icon: "📖" },
  { key: "praxias", label: "Gimnasia", icon: "🤸" },
];

// Decorado de las pantallas del niño: cielo, nubes y colina con flores.
function PantallaNinos({ children }) {
  return (
    <div className="tab-shell ninos-shell">
      <div className="cielo" aria-hidden="true">
        <span className="sol">☀️</span>
        <span className="nube n1">☁️</span>
        <span className="nube n2">☁️</span>
        <span className="nube n3">☁️</span>
        <span className="colina" />
        <span className="flor" style={{ left: "8%" }}>🌷</span>
        <span className="flor" style={{ left: "30%", animationDelay: "-1s" }}>🌼</span>
        <span className="flor" style={{ left: "62%", animationDelay: "-2s" }}>🌻</span>
        <span className="flor" style={{ left: "88%", animationDelay: "-.5s" }}>🌷</span>
      </div>
      <div className="tab-content">{children}</div>
    </div>
  );
}

export default function App() {
  const [index, setIndex] = useState(0);
  // Las pantallas con cámara piden el micrófono/cámara apenas se montan: no se montan hasta que el niño toca esa pestaña.
  const [visited, setVisited] = useState(new Set([0]));
  // "inicio" = recorrido de entrada (hola, nombre, test previo). "app" = Aprender y Gimnasia.
  // Rutas: "#/" es la entrada y "#/aprender" es la app; así "atrás" y recargar llevan donde corresponde.
  // Mientras no haya hecho el test previo SIEMPRE se empieza por la entrada, sin importar la dirección que tenga el navegador.
  // También se pide el nombre del niño: sin nombre guardado siempre se empieza por "¡Hola!" y su nombre.
  const rutaDeHash = () => (testPrevioHecho() && getNombre() ? "app" : "inicio");
  const [modo, setModoEstado] = useState(rutaDeHash);
  const [jugando, setJugando] = useState(null); // clave del sonido que se está practicando (pantalla de juego)
  const setModo = (m) => {
    setModoEstado(m);
    const hash = m === "app" ? "#/aprender" : "#/";
    if (window.location.hash !== hash) window.history.pushState(null, "", hash);
  };
  useEffect(() => {
    const alCambiar = () => setModoEstado(rutaDeHash());
    window.addEventListener("popstate", alCambiar);
    window.addEventListener("hashchange", alCambiar);
    return () => { window.removeEventListener("popstate", alCambiar); window.removeEventListener("hashchange", alCambiar); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function selectTab(i) {
    setIndex(i);
    setVisited((prev) => new Set(prev).add(i));
  }

  // Practicando un sonido: pantalla del niño, con la flecha ← de vuelta al camino.
  if (jugando) {
    return (
      <PantallaNinos>
        <JugarScreen key={jugando} claveInicial={jugando} onSalir={() => setJugando(null)} />
      </PantallaNinos>
    );
  }

  // Entrada: hola, nombre, cómo se juega y test previo. Al terminar pasa a Aprender.
  if (modo === "inicio") {
    return (
      <PantallaNinos>
        <JugarScreen onAprender={() => { setIndex(0); setModo("app"); }} />
      </PantallaNinos>
    );
  }

  return (
    <div className="tab-shell">
      <div className="bottom-nav-wrap">
        <div className="bottom-nav">
          <div className="sidebar-logo">
            <span className="sidebar-logo-icon">🗣️</span>
            <span className="sidebar-logo-text">FonoKids</span>
          </div>
          {TABS.map((tab, i) => (
            <button
              key={tab.key}
              className={`bottom-nav-item ${index === i ? "selected" : ""}`}
              onClick={() => selectTab(i)}
            >
              <span className="bottom-nav-icon">{tab.icon}</span>
              <span className="bottom-nav-label">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="tab-content">
        <div className="tab-pane" style={{ display: index === 0 ? "flex" : "none" }}>
          {visited.has(0) && <LearningPathScreen onJugar={(clave, modo) => setJugando(modo ? `${clave}|${modo}` : clave)} />}
        </div>
        <div className="tab-pane" style={{ display: index === 1 ? "flex" : "none" }}>
          {visited.has(1) && index === 1 && <LetrasScreen />}
        </div>
        <div className="tab-pane" style={{ display: index === 2 ? "flex" : "none" }}>
          {visited.has(2) && index === 2 && <EjerciciosScreen />}
        </div>
      </div>
    </div>
  );
}
