import { useEffect, useMemo, useRef, useState } from "react";
import "./MouthAvatar.css";

// 25fps -- misma cadencia que graba grabar_video_continuo.py, así la
// animación se ve a la velocidad real de habla, ni acelerada ni en cámara
// lenta.
const FPS = 25;
const FRAME_MS = 1000 / FPS;

/// Boquita animada a partir de los 20 puntos de labios REALES (índices
/// iBUG 48-67: 0-11 contorno externo, 12-19 contorno interno) de un clip
/// "<palabra>_correcto" -- ver GET /reference/<palabra> en server/main.py.
/// No es una animación inventada: son las coordenadas normalizadas que el
/// propio modelo usa para aprender, solo que acá se dibujan en vez de
/// alimentar una red neuronal.
export default function MouthAvatar({ frames, size = 220 }) {
  const [frameIdx, setFrameIdx] = useState(0);
  const rafRef = useRef(null);

  // Bounding box FIJO calculado sobre TODOS los frames (no por-frame) --
  // si se recalculara cuadro a cuadro, el zoom "respiraría" cada vez que
  // los labios ocupan más o menos espacio, en vez de quedar estable.
  const viewBox = useMemo(() => {
    if (!frames || frames.length === 0) return { minX: -1, minY: -1, w: 2, h: 2 };
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const frame of frames) {
      for (const [x, y] of frame) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const marginRatio = 0.35; // aire alrededor, si no los labios tocan el borde
    const w = (maxX - minX) * (1 + marginRatio) || 1;
    const h = (maxY - minY) * (1 + marginRatio) || 1;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    return { minX: cx - w / 2, minY: cy - h / 2, w, h };
  }, [frames]);

  useEffect(() => {
    if (!frames || frames.length === 0) return undefined;
    setFrameIdx(0);

    // setTimeout en vez de requestAnimationFrame: acá solo necesitamos una
    // cadencia fija de 25fps (no sincronía con el refresh de pantalla), y
    // rAF no dispara en algunos entornos de automatización/headless.
    function tick() {
      setFrameIdx((i) => (i + 1) % frames.length);
      rafRef.current = setTimeout(tick, FRAME_MS);
    }
    rafRef.current = setTimeout(tick, FRAME_MS);
    return () => clearTimeout(rafRef.current);
  }, [frames]);

  if (!frames || frames.length === 0) return null;

  const points = frames[frameIdx];
  const outer = points.slice(0, 12);
  const inner = points.slice(12, 20);
  const toPath = (pts) => pts.map(([x, y]) => `${x},${y}`).join(" ");

  return (
    <svg
      className="mouth-avatar"
      width={size}
      height={size}
      viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.w} ${viewBox.h}`}
    >
      <polygon points={toPath(outer)} className="mouth-avatar-outer" />
      <polygon points={toPath(inner)} className="mouth-avatar-inner" />
    </svg>
  );
}
