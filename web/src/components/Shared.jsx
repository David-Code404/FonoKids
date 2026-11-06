// Piezas compartidas por Historial/Palabras -- equivalentes a AppBackground,
// AppCard, HeroCard, ScreenHeader, SectionLabel, ErrorState y EmptyState de
// app_theme.dart, para que las pantallas compartan el mismo lenguaje visual.
import "./Shared.css";

export function AppBackground({ children }) {
  return (
    <div className="app-background">
      <div className="glow-blob glow-1" />
      <div className="glow-blob glow-2" />
      <div className="app-background-content">{children}</div>
    </div>
  );
}

export function AppCard({ children, padded = true }) {
  return <div className={`app-card ${padded ? "app-card-padded" : ""}`}>{children}</div>;
}

export function HeroCard({ children }) {
  return <div className="hero-card">{children}</div>;
}

export function ScreenHeader({ title, subtitle, icon = "🗣", onAction, actionIcon = "↻" }) {
  return (
    <div className="screen-header">
      <div className="screen-header-icon">{icon}</div>
      <div className="screen-header-text">
        <div className="screen-header-title">{title}</div>
        <div className="screen-header-subtitle">{subtitle}</div>
      </div>
      {onAction && (
        <button className="screen-header-action" onClick={onAction}>
          {actionIcon}
        </button>
      )}
    </div>
  );
}

export function SectionLabel({ children }) {
  return <div className="section-label">{children}</div>;
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="state-message">
      <div className="state-icon">☁✕</div>
      <div className="state-text">{message}</div>
      <button className="state-retry" onClick={onRetry}>
        ↻ Reintentar
      </button>
    </div>
  );
}

export function EmptyState({ icon = "○", message }) {
  return (
    <div className="state-message">
      <div className="state-icon">{icon}</div>
      <div className="state-text">{message}</div>
    </div>
  );
}
