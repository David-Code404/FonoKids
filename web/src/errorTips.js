// Nombre lindo + ícono + consejo de práctica por subtipo de error --
// mismos nombres que devuelve el servidor (server/main.py,
// _parse_clase_predicha). Fuente única compartida entre ErrorsScreen (Para
// practicar) y HomeScreen (feedback al toque después de cada intento), para
// que el chico vea el mismo lenguaje en los dos lados: no solo "está mal",
// sino DÓNDE exactamente le salió mal y qué probar la próxima vez.
export const TIPO_ERROR_INFO = {
  lambdacismo: {
    label: "Dijo L en vez de R",
    icon: "👅",
    tip: "Probá subir la lengua rapidito contra el paladar y separarla enseguida, sin dejarla pegada arriba.",
  },
  dentalizacion: {
    label: "Lengua contra los dientes",
    icon: "🦷",
    tip: "La lengua te tocó los dientes de adelante -- probá tocar más atrás, contra el paladar.",
  },
  omision: {
    label: "No hizo el sonido",
    icon: "🤐",
    tip: "No se escuchó el sonido -- probá marcarlo bien fuerte esta vez, sin comértelo.",
  },
};

export function tipoErrorInfo(tipo) {
  return TIPO_ERROR_INFO[tipo] || { label: "Para seguir practicando", icon: "🔁", tip: "Probemos de nuevo, despacito." };
}
