import { Component } from "react";

/// Si algo revienta en React (ej. un bug de HMR con código desactualizado,
/// como pasó antes con un closure viejo), ANTES la pantalla se quedaba
/// muda -- sin grabar, sin predecir, sin ningún aviso. Con esto en vez de
/// eso se ve un error claro con botón de recargar, para que nunca quede
/// "fallando en silencio" sin que el usuario sepa qué pasó.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Error atrapado por ErrorBoundary:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            height: "100vh",
            width: "100vw",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            background: "#fff8ec",
            color: "#3a3550",
            fontFamily: "'Baloo 2', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            padding: 24,
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 40 }}>⚠</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>Algo se rompió en la app</div>
          <div style={{ fontSize: 13, color: "#6b6580", maxWidth: 340 }}>
            {this.state.error.message || String(this.state.error)}
          </div>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: "10px 20px",
              borderRadius: 12,
              border: "none",
              background: "#8c6bff",
              color: "#fff",
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Recargar
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
