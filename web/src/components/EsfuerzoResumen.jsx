import { ETIQUETAS_ESFUERZO } from "../ejercicios.js";

/// Estrellas por indicador de esfuerzo (1 a 3); solo se muestran los que se
/// midieron. Nunca porcentajes: es para el chico (ver estimación del esfuerzo
/// en ejercicios.js).
export default function EsfuerzoResumen({ esfuerzo }) {
  if (!esfuerzo) return null;
  const claves = Object.keys(ETIQUETAS_ESFUERZO).filter((k) => esfuerzo[k]);
  if (claves.length === 0) return null;
  return (
    <div className="esfuerzo-resumen" style={{ display: "grid", gap: 6, margin: "8px 0" }}>
      {claves.map((k) => (
        <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <span>{ETIQUETAS_ESFUERZO[k]}</span>
          <span aria-label={`${esfuerzo[k]} de 3`}>{"⭐".repeat(esfuerzo[k])}{"☆".repeat(3 - esfuerzo[k])}</span>
        </div>
      ))}
    </div>
  );
}
