import { useCallback, useEffect, useRef, useState } from "react";
import { getServerUrl } from "../api.js";
import "./EtiquetarScreen.css";

// Pantalla para el docente / los papás (dirección #/etiquetar): se oye cada grabación y se marca lo que de verdad se dijo.
// Con esas etiquetas se mide cuánto acierta el sistema y se puede afinar el modelo de audio.
async function pedir(ruta, opciones) {
  const r = await fetch(`${getServerUrl()}${ruta}`, opciones);
  if (!r.ok) throw new Error(`Error ${r.status}`);
  return r.json();
}

export default function EtiquetarScreen() {
  const [lista, setLista] = useState([]);
  const [conteo, setConteo] = useState({ total: 0, etiquetados: 0 });
  const [resumen, setResumen] = useState(null);
  const [error, setError] = useState("");
  const [cargando, setCargando] = useState(true);
  const audioRef = useRef(null);

  const cargar = useCallback(async () => {
    try {
      const [p, r] = await Promise.all([pedir("/etiquetar/pendientes?limite=50"), pedir("/etiquetar/resumen")]);
      setLista(p.pendientes);
      setConteo({ total: p.total, etiquetados: p.etiquetados });
      setResumen(r);
      setError("");
    } catch (e) {
      setError("No pude hablar con el servidor. ¿Está prendido?");
    } finally {
      setCargando(false);
    }
  }, []);
  useEffect(() => { cargar(); }, [cargar]);

  const actual = lista[0];
  useEffect(() => { if (actual && audioRef.current) { audioRef.current.load(); audioRef.current.play().catch(() => {}); } }, [actual?.archivo]); // eslint-disable-line react-hooks/exhaustive-deps

  async function marcar(etiqueta) {
    if (!actual) return;
    try {
      await pedir("/etiquetar", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ archivo: actual.archivo, etiqueta }) });
      setLista((l) => l.slice(1));
      setConteo((c) => ({ ...c, etiquetados: c.etiquetados + 1 }));
      if (lista.length <= 1) cargar();
      else pedir("/etiquetar/resumen").then(setResumen).catch(() => {});
    } catch {
      setError("No se pudo guardar la etiqueta. Prueba otra vez.");
    }
  }

  return (
    <div className="etq-pantalla">
      <h1 className="etq-titulo">🎧 Etiquetar grabaciones</h1>
      <p className="etq-ayuda">Escucha cada grabación y marca si la palabra se dijo <b>bien</b> o <b>mal</b> (no lo que opina el sistema, sino lo que tú oyes).</p>
      <div className="etq-contador">Etiquetadas: <b>{conteo.etiquetados}</b> de {conteo.total} · Meta para afinar el modelo: 200 a 300</div>
      {error && <div className="etq-error">{error}</div>}
      {cargando ? <p>Cargando…</p> : actual ? (
        <div className="etq-tarjeta">
          <div className="etq-palabra">{actual.palabra}</div>
          <audio ref={audioRef} controls src={`${getServerUrl()}/etiquetar/audio?archivo=${encodeURIComponent(actual.archivo)}`} className="etq-audio" />
          <div className="etq-botones">
            <button className="etq-btn etq-bien" onClick={() => marcar("bien")}>✅ Bien dicho</button>
            <button className="etq-btn etq-mal" onClick={() => marcar("mal")}>❌ Mal dicho</button>
            <button className="etq-btn etq-duda" onClick={() => marcar("dudoso")}>❓ No se entiende</button>
          </div>
          <div className="etq-quedan">Quedan {lista.length}{lista.length === 50 ? "+" : ""} por etiquetar</div>
        </div>
      ) : <div className="etq-tarjeta"><p>🎉 No hay grabaciones pendientes. Se irán sumando cuando los niños usen la app.</p></div>}

      {resumen && resumen.comparados > 0 && (
        <div className="etq-resumen">
          <h2>📊 Qué tan bien acierta el sistema</h2>
          <p>Coincidió con tu opinión en <b>{resumen.acuerdo_pct}%</b> de {resumen.comparados} grabaciones.</p>
          <ul>
            <li>Bien dicho y el sistema dijo bien: {resumen.bien_bien}</li>
            <li>Mal dicho y el sistema dijo mal: {resumen.mal_mal}</li>
            <li><b>Falsos aprobados</b> (era mal y dijo bien): {resumen.falso_aprobado}</li>
            <li><b>Falsos rechazos</b> (era bien y dijo mal): {resumen.falso_rechazo}</li>
            <li>No se entendía: {resumen.dudoso}</li>
          </ul>
          {Object.keys(resumen.por_palabra).length > 0 && (
            <table className="etq-tabla">
              <thead><tr><th>Palabra</th><th>Grabaciones</th><th>Acuerdo</th></tr></thead>
              <tbody>
                {Object.entries(resumen.por_palabra).sort((a, b) => b[1].total - a[1].total).map(([p, d]) => (
                  <tr key={p}><td>{p}</td><td>{d.total}</td><td>{Math.round((100 * d.acuerdo) / d.total)}%</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
