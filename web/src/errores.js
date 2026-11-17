// Explica al niño, con palabras sencillas y dibujos, QUÉ salió mal en su intento y QUÉ hacer para corregirlo.
import { fonemasDe, sonidosAMejorar, pistaCara } from "./fonemas.js";

// Cómo se escriben, para un niño, los sonidos que reconoce el servidor (alfabeto fonético -> letras).
const IPA = {
  "θ": "z", "ð": "d", "ɾ": "r", "r": "rr", "x": "j", "ɡ": "g", "ʝ": "y", "ʎ": "ll", "tʃ": "ch", "ɲ": "ñ", "ŋ": "n",
  "β": "b", "ɣ": "g", "ɛ": "e", "ɔ": "o", "ɪ": "i", "ʊ": "u", "ɑ": "a", "ə": "e", "ɐ": "a", "ʌ": "o", "ø": "o", "æ": "a",
  "ɚ": "e", "ʃ": "ch", "ʒ": "y", "dʒ": "y", "ɟ": "y", "ɫ": "l", "ɹ": "r", "ʁ": "r",
};

/// "k a d o" (lo que oyó el servidor) -> "cado" (para mostrárselo al niño). Devuelve "" si no se puede.
export function oidoComoLetras(libre) {
  if (!libre || typeof libre !== "string") return "";
  const letras = libre.split(/\s+/).map((t) => {
    const limpio = t.replace(/ː/g, "");
    if (IPA[limpio]) return IPA[limpio];
    if (/^[a-zñ]$/.test(limpio)) return limpio === "k" ? "c" : limpio === "v" ? "b" : limpio;
    return IPA[limpio[0]] || "";
  });
  const texto = letras.join("");
  return texto.length >= 2 && texto.length <= 14 ? texto : "";
}

const QUE_PASO_R = {
  omision: "La R se perdió: no se oyó.",
  lambdacismo: "La R sonó como una L: la lengua se quedó pegada.",
  dentalizacion: "La R sonó como una D: la lengua golpeó los dientes.",
};

/// Qué salió mal en un intento de decir una palabra. Devuelve { lineas: [{ icono, texto }], voz }.
/// `ctx`: { pico } volumen más alto de la grabación (0 a 1) y { bocaCerrada } si la cámara vio la boca casi quieta.
export function explicarError(palabra, resultado, ctx = {}) {
  const lineas = [];
  const voz = [];
  const pico = ctx.pico ?? 1;

  if (ctx.bocaCerrada) {
    lineas.push({ icono: "👄", texto: "Casi no moviste los labios." });
    lineas.push({ icono: "👉", texto: "¡Mueve los labios y abre bien la boca!" });
    voz.push("Casi no moviste los labios. ¡Mueve los labios y abre bien la boca!");
    return { lineas, voz: voz.join(" ") };
  }

  if (ctx.camaraMal) {
    lineas.push({ icono: "📷", texto: "La cámara vio que los labios no hicieron bien el movimiento." });
    lineas.push({ icono: "👉", texto: "Mira el video y mueve los labios igual." });
    voz.push("La cámara vio que los labios no hicieron bien el movimiento. Mira el video y mueve los labios igual.");
  }

  if (pico < 0.05) {
    lineas.push({ icono: "🔊", texto: "Se oyó muy bajito." });
    lineas.push({ icono: "👉", texto: "Habla más fuerte y cerca del micrófono." });
    voz.push("Se oyó muy bajito. Habla más fuerte y cerca del micrófono.");
  }

  const oido = oidoComoLetras(resultado?.reconocido_libre);
  if (oido && oido !== palabra.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")) {
    lineas.push({ icono: "👂", texto: `Yo escuché: «${oido}»` });
    voz.push(`Yo escuché ${oido}.`);
  }

  const unidades = fonemasDe(palabra);
  const malas = sonidosAMejorar(palabra, resultado?.scores) || [];
  for (const m of malas.slice(0, 2)) {
    const u = unidades[m.indice];
    const esR = u && (u.sonido === "r" || u.sonido === "rr");
    let texto;
    if (esR && resultado?.tipo_error && QUE_PASO_R[resultado.tipo_error]) texto = QUE_PASO_R[resultado.tipo_error];
    else if (m.score < -7) texto = `La ${m.letras} casi no se oyó.`;
    else texto = `La ${m.letras} no sonó bien.`;
    lineas.push({ icono: "❌", texto });
    voz.push(texto);
  }

  if (!malas.length && !lineas.length) {
    lineas.push({ icono: "❌", texto: `Algo no sonó como «${palabra}».` });
    voz.push(`Algo no sonó como ${palabra}.`);
  }

  // Cómo se dice: la guía de la letra que falló. Si no se pudo ubicar cuál fue, la de menor puntaje o la primera consonante.
  let primera = malas[0] ? unidades[malas[0].indice] : null;
  if (!primera && unidades.length) {
    const sc = Array.isArray(resultado?.scores) && resultado.scores.length === unidades.length ? resultado.scores : null;
    let k = unidades.findIndex((u) => !/^[aeiou]$/.test(u.sonido));
    if (sc) sc.forEach((v, j) => { if (v != null && (sc[k] == null || v < sc[k])) k = j; });
    primera = unidades[Math.max(k, 0)];
  }
  const pc = primera ? pistaCara(primera.sonido) : null;
  if (pc) {
    voz.push(`${pc.parte} ${pc.texto}`);
    // En el test no hay video de ayuda: la guía de cómo se dice va escrita en la tarjeta.
    if (ctx.conPista) {
      const [icono, ...resto] = pc.parte.split(" ");
      lineas.push({ icono, texto: `Así se hace la ${primera.letras}: ${resto.join(" ").toLowerCase()} ${pc.texto}` });
    }
  }
  return { lineas, voz: voz.join(" ") };
}

/// Cuando no se oyó voz (murmullo, boca cerrada, silencio o ruido): qué hacer.
export function explicarSinVoz(ctx = {}) {
  if (ctx.bocaQuieta) {
    return {
      lineas: [
        { icono: "👄", texto: "Casi no moviste los labios." },
        { icono: "👉", texto: "¡Mueve los labios y abre bien la boca!" },
        { icono: "🔊", texto: "Y habla fuerte y claro." },
      ],
      voz: "Casi no moviste los labios. Mueve los labios, abre bien la boca y habla fuerte y claro.",
    };
  }
  return {
    lineas: [
      { icono: "👂", texto: "No te oí bien." },
      { icono: "👄", texto: "Abre bien la boca." },
      { icono: "🔊", texto: "Habla fuerte y claro." },
    ],
    voz: "No te oí bien. Abre bien la boca y habla fuerte y claro.",
  };
}
