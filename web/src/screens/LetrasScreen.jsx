import { useEffect, useState } from "react";
import { ejemploDe, pistaCara, decirSonido, videosDe } from "../fonemas.js";
import { VideoReferencia } from "./SonidosScreen.jsx";
import { hablar, hablarUnaVez } from "../voz.js";
import { useAyuda } from "../useAyuda.js";
import Mascota from "../components/Mascota.jsx";
import "./LetrasScreen.css";

// "Cómo se dice": todas las letras y sonidos del español en orden, cada uno con su dibujo de ejemplo, la parte de la cara
// que se usa y qué hacer. Tocar una tarjeta hace sonar el sonido y Fofo explica cómo se hace.
const GRUPOS = [
  { titulo: "Las vocales", color: "rosa", letras: [["A", "a"], ["E", "e"], ["I", "i"], ["O", "o"], ["U", "u"]] },
  { titulo: "Con los labios", color: "morado", letras: [["M", "m"], ["P", "p"], ["B / V", "b"], ["W", "w"]] },
  { titulo: "Con los dientes", color: "verde", letras: [["F", "f"], ["S", "s"], ["Z", "z"]] },
  { titulo: "Con la lengua arriba", color: "naranja", letras: [["T", "t"], ["D", "d"], ["N", "n"], ["L", "l"], ["R", "r"], ["RR", "rr"], ["LL / Y", "ll"], ["Ñ", "ñ"], ["CH", "ch"]] },
  { titulo: "Con la lengua atrás y la garganta", color: "azul", letras: [["C / K / Q", "k"], ["G", "g"], ["J", "j"], ["X", "ks"]] },
];

function Tarjeta({ nombre, sonido, color }) {
  const [video, setVideo] = useState(false);
  const pc = pistaCara(sonido);
  const ej = ejemploDe(sonido);
  const videos = videosDe(sonido);
  if (!pc) return null;
  const [icono, ...resto] = pc.parte.split(" ");
  const explicar = () => { decirSonido(sonido); setTimeout(() => hablar(`${resto.join(" ")} ${pc.texto}`), 900); };
  return (
    <div className={`letra-tarjeta letra-${color}`}>
      <button className="letra-principal" onClick={explicar} aria-label={`Escuchar la ${nombre}`}>
        <span className="letra-grande">{nombre}</span>
        <span className="letra-sonido">🔊</span>
      </button>
      {ej && (
        <div className="letra-ejemplo"><span>{ej.dibujo}</span><b>{ej.palabra}</b></div>
      )}
      <div className="letra-como"><span className="letra-parte">{icono}</span><span>{resto.join(" ")} {pc.texto}</span></div>
      {videos.length > 0 && (
        <div className="letra-video">
          {video ? <VideoReferencia src={videos[0]} etiqueta={`Mira cómo se dice la ${nombre}`} /> : (
            <button className="letra-ver" onClick={() => setVideo(true)}>▶ Mira cómo se dice</button>
          )}
        </div>
      )}
    </div>
  );
}

const GUIA = "Toca una letra y escucha cómo suena. Después toca el botón Mira cómo se dice, y copia lo que hace la boca.";

export default function LetrasScreen() {
  // Guía hablada: al entrar Fofo explica cómo usar la pantalla (una vez) y, si el niño se queda quieto, la repite.
  useEffect(() => {
    const t = setTimeout(() => hablarUnaVez("letras-guia", GUIA), 600);
    return () => clearTimeout(t);
  }, []);
  useAyuda("Toca una letra para escucharla.");
  return (
    <div className="path-screen letras-pantalla">
      <div className="path-header">
        <span className="path-title">📖 Cómo se dice</span>
      </div>
      <div className="letras-contenido">
        <div className="letras-guia">
          <Mascota mensaje="¡Aprende cómo se dice cada letra!" />
          <ol className="letras-pasos">
            <li><em>1</em><span>👆</span>Toca una letra y escucha</li>
            <li><em>2</em><span>▶️</span>Toca "Mira cómo se dice"</li>
            <li><em>3</em><span>👄</span>Copia lo que hace la boca</li>
          </ol>
        </div>
        <p className="letras-intro">Toca una letra para oír cómo suena y ver qué parte de la cara usar. 👄 labios · 👅 lengua · 🦷 dientes · 🗣️ garganta</p>
        {GRUPOS.map((g) => (
          <section key={g.titulo} className="letras-grupo">
            <h3 className={`letras-titulo letra-${g.color}`}>{g.titulo}</h3>
            <div className="letras-rejilla">
              {g.letras.map(([nombre, sonido]) => <Tarjeta key={sonido} nombre={nombre} sonido={sonido} color={g.color} />)}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
