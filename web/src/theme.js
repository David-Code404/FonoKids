// Paleta compartida por toda la app -- misma que app_theme.dart (AppColors)
// para que la versión web se vea igual que la app Flutter original.
export const colors = {
  background: "#07080D",
  surface: "#15171F",
  surfaceAlt: "#1C1F29",
  border: "#272A35",
  accent: "#7C6CFF",
  accent2: "#35C9C1",
  accent3: "#FFB454",
  danger: "#FF5C7A",
  textPrimary: "#F2F3F7",
  textSecondary: "#9DA2B3",
  textMuted: "#5D6273",
};

export const chartPalette = [
  colors.accent,
  colors.accent2,
  colors.accent3,
  "#FF6B81",
  "#5AA9FF",
];

export const heroGradient = `linear-gradient(135deg, #6A5CFF, ${colors.accent2})`;
