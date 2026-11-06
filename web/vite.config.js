import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// host: "0.0.0.0" para poder abrir la app desde el celular en la misma red
// WiFi (ej. http://<ip-de-esta-pc>:5173), igual que el server FastAPI en
// server/main.py escucha en 0.0.0.0:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Nunca servir nada cacheado en desarrollo -- si el navegador (sobre
    // todo el del celular) guarda una versión vieja del JS, podés terminar
    // grabando contra código desactualizado con bugs ya arreglados, y
    // encima sin ningún error visible (justo lo que pasó antes con un HMR
    // viejo). Mejor forzar que siempre pida la versión actual.
    headers: {
      "Cache-Control": "no-store",
    },
  },
});
