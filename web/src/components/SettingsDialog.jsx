import { useState } from "react";
import { checkHealth } from "../api.js";
import "./SettingsDialog.css";

/// Diálogo para configurar la URL del servidor (server/main.py) -- igual
/// que settings_dialog.dart, con botón "Probar conexión".
export default function SettingsDialog({ currentUrl, onSave, onClose }) {
  const [value, setValue] = useState(currentUrl);
  const [testing, setTesting] = useState(false);
  const [testOk, setTestOk] = useState(null);

  async function testConnection() {
    setTesting(true);
    setTestOk(null);
    const ok = await checkHealth(value);
    setTesting(false);
    setTestOk(ok);
  }

  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-title">Servidor de predicción</div>
        <p className="dialog-hint">
          IP y puerto de la PC que corre server/main.py (debe estar en la misma red WiFi que el
          dispositivo).
        </p>
        <input
          className="dialog-input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="http://192.168.1.100:8000"
        />
        <div className="dialog-test-row">
          <button className="dialog-test-button" onClick={testConnection} disabled={testing}>
            {testing ? "Probando..." : "📶 Probar conexión"}
          </button>
          {testOk === true && <span className="dialog-test-ok">✓</span>}
          {testOk === false && <span className="dialog-test-fail">⚠</span>}
        </div>
        <div className="dialog-actions">
          <button className="dialog-button ghost" onClick={onClose}>
            Cancelar
          </button>
          <button className="dialog-button primary" onClick={() => onSave(value)}>
            Guardar
          </button>
        </div>
      </div>
    </div>
  );
}
