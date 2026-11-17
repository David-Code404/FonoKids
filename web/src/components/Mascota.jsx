import { useEffect, useState } from "react";
import { alCambiarVoz, hablar, ponerVoz, vozActiva } from "../voz.js";
import "./Mascota.css";

// Botón 🔊/🔇 para prender o apagar la voz de la mascota.
export function BotonVoz() {
  const [activa, setActiva] = useState(vozActiva());
  useEffect(() => alCambiarVoz(setActiva), []);
  return (
    <button
      className="boton-voz"
      onClick={() => { const n = !activa; ponerVoz(n); if (n) hablar("¡Aquí estoy!", { forzar: true }); }}
      aria-label={activa ? "Silenciar la voz" : "Activar la voz"}
      title={activa ? "Silenciar la voz" : "Activar la voz"}
    >
      {activa ? "🔊" : "🔇"}
    </button>
  );
}

// La mascota de FonoKids: un zorrito que habla con globito. Al tocarla repite lo que dijo.
export default function Mascota({ mensaje, animo = "feliz", grande = false }) {
  const cara = { feliz: "🦊", pensando: "🦊", animo: "🦊", festejo: "🦊" }[animo] || "🦊";
  return (
    <div className={`mascota ${grande ? "mascota-grande" : ""} mascota-${animo}`}>
      <button className="mascota-cara" onClick={() => hablar(mensaje, { forzar: true })} aria-label="Escuchar a Fofo">
        <span>{cara}</span>
        {animo === "festejo" ? <i>🎉</i> : animo === "animo" ? <i>💪</i> : animo === "pensando" ? <i>💭</i> : null}
      </button>
      {animo === "festejo" && (
        <span className="confeti" aria-hidden="true"><i>🎉</i><i>⭐</i><i>✨</i><i>🎊</i><i>⭐</i><i>✨</i></span>
      )}
      {mensaje && <div className="mascota-globo">{mensaje}</div>}
    </div>
  );
}
