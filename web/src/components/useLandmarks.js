import { useEffect } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Puntos de la cara EN VIVO sobre el video (mismos puntos y colores que la práctica de siempre):
// carga el FaceLandmarker de MediaPipe y dibuja en un <canvas> encima del <video>.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const LABIOS = new Set([
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78,
  95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]);
const OJOS = new Set([33, 263]);
const MEJILLAS = [205, 425, 50, 280, 116, 345, 147, 376];
const NARIZ = [168, 6, 197, 195, 5, 4, 1, 98, 327];
const MANDIBULA = [152, 148, 377, 176, 400, 149, 378, 150, 379];

function grupos() {
  return [
    { c: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, color: "rgba(124,92,255,0.85)", ancho: 2.5 },
    { c: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, color: "rgba(255,176,32,0.9)", ancho: 2 },
    { c: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, color: "rgba(255,176,32,0.9)", ancho: 2 },
    { c: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, color: "rgba(80,200,255,0.9)", ancho: 1.5 },
    { c: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, color: "rgba(80,200,255,0.9)", ancho: 1.5 },
    { c: FaceLandmarker.FACE_LANDMARKS_LIPS, color: "rgba(46,204,143,0.95)", ancho: 2 },
  ];
}

function puntos(ctx, cara, indices, color, radio, w, h) {
  ctx.fillStyle = color;
  for (const i of indices) {
    const p = cara[i];
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radio, 0, Math.PI * 2);
    ctx.fill();
  }
}

/// Dibuja los puntos de la cara sobre `canvasRef` mientras `activo` sea true.
export function useLandmarks(videoRef, canvasRef, activo) {
  useEffect(() => {
    if (!activo) return undefined;
    let cancelado = false;
    let landmarker = null;
    let raf = null;
    let n = 0;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const lm = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
        });
        if (cancelado) { lm.close(); return; }
        landmarker = lm;
      } catch (e) {
        console.warn("No se pudo cargar el detector de la cara:", e);
      }
    })();

    function loop() {
      raf = requestAnimationFrame(loop);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !landmarker || video.readyState < 2) return;
      n += 1;
      if (n % 2 !== 0) return;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cara = landmarker.detectForVideo(video, performance.now()).faceLandmarks?.[0];
      if (!cara) return;
      const w = canvas.width;
      const h = canvas.height;
      ctx.lineCap = "round";
      for (const g of grupos()) {
        ctx.strokeStyle = g.color;
        ctx.lineWidth = g.ancho;
        ctx.beginPath();
        for (const { start, end } of g.c) {
          const a = cara[start];
          const b = cara[end];
          if (!a || !b) continue;
          ctx.moveTo(a.x * w, a.y * h);
          ctx.lineTo(b.x * w, b.y * h);
        }
        ctx.stroke();
      }
      puntos(ctx, cara, LABIOS, "#2ECC8F", 2.5, w, h);
      puntos(ctx, cara, MANDIBULA, "#7C5CFF", 3.5, w, h);
      puntos(ctx, cara, MEJILLAS, "#FF6FA5", 4, w, h);
      puntos(ctx, cara, NARIZ, "#FFFFFF", 2.5, w, h);
      puntos(ctx, cara, OJOS, "#FFD84D", 3.5, w, h);
    }
    raf = requestAnimationFrame(loop);

    return () => {
      cancelado = true;
      if (raf) cancelAnimationFrame(raf);
      landmarker?.close();
    };
  }, [activo, videoRef, canvasRef]);
}
