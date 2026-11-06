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
  anteriorizacion: {
    label: "La lengua se adelantó",
    icon: "🦷",
    tip: "La lengua tiene que tocar más atrás, cerca de la garganta -- probá no adelantarla hasta los dientes.",
  },
  debilitamiento: {
    label: "Salió muy suave",
    icon: "💨",
    tip: "El sonido salió como un soplido flojo -- probá marcarlo con un golpe más firme.",
  },
  interdental: {
    label: "La lengua se asomó entre los dientes",
    icon: "👅",
    tip: "Probá dejar la lengua detrás de los dientes de arriba, sin sacarla.",
  },
  oclusion: {
    label: "Se tapó el sonido",
    icon: "🚫",
    tip: "El sonido salió tapado -- probá dejar pasar el aire suave en vez de cortarlo.",
  },
  epentesis: {
    label: "Metió una vocal de más",
    icon: "➕",
    tip: "Se coló una vocal en el medio -- probá decir las dos consonantes juntas, sin nada en el medio.",
  },
  distorsion: {
    label: "El sonido salió distinto",
    icon: "🔀",
    tip: "Ese sonido salió un poco distinto -- probá repetirlo despacito, marcando bien cada parte.",
  },
  sustitucion_acustica: {
    label: "El sonido no salió claro",
    icon: "🔊",
    tip: "Aunque la boca se vio bien, el sonido no salió como tenía que sonar -- probá decirlo un poco más fuerte y claro.",
  },
  motor_severo: {
    label: "Sonido y boca necesitan práctica",
    icon: "🧩",
    tip: "Ni el sonido ni el movimiento de la boca salieron como tenían que salir -- vamos despacito, paso a paso.",
  },
  compensacion_visual: {
    label: "La boca no acompañó al sonido",
    icon: "👄",
    tip: "El sonido salió bien, pero la boca se movió distinto de lo esperado -- probá exagerar un poco el movimiento de los labios.",
  },
};

export function tipoErrorInfo(tipo) {
  return TIPO_ERROR_INFO[tipo] || { label: "Para seguir practicando", icon: "🔁", tip: "Probemos de nuevo, despacito." };
}
