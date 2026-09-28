import { useState } from "react";
import PhrasesScreen from "./screens/PhrasesScreen.jsx";
import ErrorsScreen from "./screens/ErrorsScreen.jsx";
import LearningPathScreen from "./screens/LearningPathScreen.jsx";
import SoundsScreen from "./screens/SoundsScreen.jsx";
import "./App.css";

// Pedido explícito: Aprender (camino de etapas, termina en la práctica real
// con cámara) / Sonidos (sonidos sueltos, sin depender de una palabra) /
// Logros (lo que salió BIEN) / Para practicar (lo que salió mal, separado
// de Logros para no mezclar estrellas con fallas). Se sacaron "Mi Diario" y
// "Práctica" como pestañas propias -- la cámara se llega siempre desde
// "Aprender" (último paso de cada etapa), no hace falta una entrada aparte.
const TABS = [
  { key: "aprender", label: "Aprender", icon: "🧭" },
  { key: "sonidos", label: "Sonidos", icon: "🔊" },
  { key: "logros", label: "Logros", icon: "🏆" },
  { key: "errores", label: "Para practicar", icon: "🔁" },
];

/// Navegación por pestañas: Aprender / Sonidos / Logros -- igual que
/// main_tab_screen.dart (MainTabScreen).
export default function App() {
  const [index, setIndex] = useState(0);
  // Ojo: las pantallas con cámara piden el micrófono/cámara apenas se
  // montan, así que NUNCA se montan hasta que el usuario realmente toca esa
  // pestaña -- igual que el Set<int> _visited en main_tab_screen.dart.
  const [visited, setVisited] = useState(new Set([0]));

  function selectTab(i) {
    setIndex(i);
    setVisited((prev) => new Set(prev).add(i));
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
          {visited.has(0) && <LearningPathScreen />}
        </div>
        <div className="tab-pane" style={{ display: index === 1 ? "flex" : "none" }}>
          {visited.has(1) && <SoundsScreen />}
        </div>
        <div className="tab-pane" style={{ display: index === 2 ? "flex" : "none" }}>
          {visited.has(2) && <PhrasesScreen active={index === 2} />}
        </div>
        <div className="tab-pane" style={{ display: index === 3 ? "flex" : "none" }}>
          {visited.has(3) && <ErrorsScreen active={index === 3} />}
        </div>
      </div>
    </div>
  );
}
