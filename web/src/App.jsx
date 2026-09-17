import { useState } from "react";
import CapturesScreen from "./screens/CapturesScreen.jsx";
import PhrasesScreen from "./screens/PhrasesScreen.jsx";
import HomeScreen from "./screens/HomeScreen.jsx";
import "./App.css";

const TABS = [
  { key: "diario", label: "Mi Diario", icon: "📔" },
  { key: "logros", label: "Logros", icon: "🏆" },
  { key: "practicar", label: "Practicar", icon: "🎥" },
];

/// Navegación por pestañas entre Mi Diario, Logros y Practicar (cámara en
/// tiempo real) -- igual que main_tab_screen.dart (MainTabScreen).
export default function App() {
  const [index, setIndex] = useState(0);
  // Ojo: HomeScreen (pestaña "Grabar") pide la cámara apenas se monta, así
  // que NUNCA se monta hasta que el usuario realmente toca esa pestaña --
  // igual que el Set<int> _visited en main_tab_screen.dart.
  const [visited, setVisited] = useState(new Set([0]));

  function selectTab(i) {
    setIndex(i);
    setVisited((prev) => new Set(prev).add(i));
  }

  return (
    <div className="tab-shell">
      <div className="tab-content">
        <div className="tab-pane" style={{ display: index === 0 ? "flex" : "none" }}>
          {visited.has(0) && <CapturesScreen />}
        </div>
        <div className="tab-pane" style={{ display: index === 1 ? "flex" : "none" }}>
          {visited.has(1) && <PhrasesScreen />}
        </div>
        <div className="tab-pane" style={{ display: index === 2 ? "flex" : "none" }}>
          {visited.has(2) && <HomeScreen />}
        </div>
      </div>

      <div className="bottom-nav-wrap">
        <div className="bottom-nav">
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
    </div>
  );
}
