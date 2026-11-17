import { useEffect } from "react";
import { hablar } from "./voz.js";

// Ayuda por inactividad: si el niño no toca nada por un rato, Fofo repite en voz alta qué hacer. SOLO UNA VEZ por pantalla
// (repetirlo muchas veces cansa). Cada toque en la pantalla reinicia la espera, pero no da más recordatorios.
export function useAyuda(mensaje, activo = true, ms = 12000) {
  useEffect(() => {
    if (!activo || !mensaje) return undefined;
    let t = null;
    let dicho = false;
    const armar = () => {
      clearTimeout(t);
      if (dicho) return;
      t = setTimeout(() => { dicho = true; hablar(mensaje); }, ms);
    };
    armar();
    window.addEventListener("pointerdown", armar);
    return () => { clearTimeout(t); window.removeEventListener("pointerdown", armar); };
  }, [mensaje, activo, ms]);
}
