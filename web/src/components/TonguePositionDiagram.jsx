import "./TonguePositionDiagram.css";

// Corte transversal simplificado de la boca (forma de "lente", vista de
// costado) -- SVG dibujado a mano, sin depender de ningún asset externo
// (sin licencias, sin motor 3D, sin GPU). Pensado para mostrar DÓNDE va la
// lengua, algo que los labios (lo único que mide el modelo visual) nunca
// pueden mostrar.

// Las 9 palabras objetivo del proyecto comparten el mismo sonido problema
// (la "r"), así que las variantes que importan de verdad son estas tres --
// el punto de contacto es parecido para las tres, lo que cambia es CÓMO
// (y por cuánto tiempo) toca la lengua ahí.
const VARIANTS = {
  vibrante: {
    label: "Vibrante (r)",
    tip: "La puntita de la lengua toca y suelta MUY rápido, como un tamborcito.",
    tongueClass: "tongue-vibrante",
  },
  lateral: {
    label: "Lateral (l) -- ojo, este es el error común",
    tip: "Acá la lengua se queda pegada arriba, no suelta rápido -- por eso 'perro' suena 'pello'.",
    tongueClass: "tongue-lateral",
  },
  dental: {
    label: "Dental (d/t) -- el otro error común",
    tip: "Acá la lengua toca los dientes de adelante, no el techito de más atrás -- por eso 'perro' suena 'pedo'.",
    tongueClass: "tongue-dental",
  },
};

// Mapea la palabra objetivo a la variante más relevante -- por ahora todas
// las palabras del proyecto trabajan la "r", así que todas apuntan a
// "vibrante" como la forma correcta (las pestañas dejan ver las otras dos
// para comparar qué error se hace).
export function articulationForWord(_word) {
  return "vibrante";
}

export default function TonguePositionDiagram({ variant = "vibrante" }) {
  const info = VARIANTS[variant] || VARIANTS.vibrante;

  return (
    <div className="tongue-diagram">
      <svg viewBox="0 0 200 140" className="tongue-diagram-svg">
        {/* Cavidad de la boca -- forma simple de "lente", de costado */}
        <path
          className="td-cavity"
          d="M18,70 C25,45 55,28 100,26 C150,24 178,45 182,70
             C178,95 150,116 100,114 C55,112 25,95 18,70 Z"
        />
        {/* Paladar (techo de la boca) */}
        <path className="td-palate" d="M35,55 C60,35 130,33 165,55" />
        {/* Úvula / campanilla, al fondo */}
        <path className="td-uvula" d="M165,55 c4,6 4,12 0,18" />
        {/* Dientes de arriba y de abajo */}
        <rect className="td-teeth" x="24" y="48" width="14" height="10" rx="3" />
        <rect className="td-teeth" x="24" y="82" width="14" height="10" rx="3" />
        {/* Labio (apertura de la boca) */}
        <path className="td-lip" d="M18,62 Q10,70 18,78" />
        {/* Punto de contacto resaltado -- donde debería tocar la lengua */}
        <circle className="td-contact-point" cx="42" cy="48" r="6" />
        {/* Lengua -- una sola forma, apoyada en el piso de la boca; la
            posición/animación cambia según variant (ver CSS). */}
        <path
          className={`td-tongue ${info.tongueClass}`}
          d="M160,95 C130,105 90,104 65,95 C50,90 42,82 40,72
             C38,85 42,96 52,104 C70,118 110,120 140,112 C152,108 158,102 160,95 Z"
        />
      </svg>
      <div className="tongue-diagram-label">{info.label}</div>
      <div className="tongue-diagram-tip">{info.tip}</div>
    </div>
  );
}
