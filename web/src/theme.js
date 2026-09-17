// Paleta compartida por toda la app -- cálida y colorida, pensada para una
// app de práctica de pronunciación para chicos (no el look oscuro tipo
// cámara de seguridad de la versión anterior). Tiene que coincidir con las
// variables CSS de index.css.
export const colors = {
  background: "#FFF8EC",
  surface: "#FFFFFF",
  surfaceAlt: "#FBF2FF",
  border: "#ECE1FB",
  accent: "#8C6BFF",
  accent2: "#2ECC8F",
  accent3: "#FFB84D",
  danger: "#FF6B6B",
  textPrimary: "#3A3550",
  textSecondary: "#6B6580",
  textMuted: "#A39CC0",
};

export const chartPalette = [
  colors.accent,
  colors.accent2,
  colors.accent3,
  "#FF8FA3",
  "#4FB3FF",
];

export const heroGradient = `linear-gradient(135deg, ${colors.accent}, ${colors.accent2})`;
