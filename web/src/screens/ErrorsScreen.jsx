import { useEffect, useState } from "react";
import { getRecordings, getServerUrl } from "../api.js";
import { iconFor } from "../wordIcons.js";
import { tipoErrorInfo } from "../errorTips.js";
import { AppCard, AppBackground, EmptyState, ScreenHeader, SectionLabel } from "../components/Shared.jsx";
import { chartPalette } from "../theme.js";
import "./Screens.css";

function aggregate(recordings) {
  const fallas = recordings.filter((r) => r.correcta === false);
  const grouped = {};
  for (const r of fallas) {
    const g = (grouped[r.word] ??= { word: r.word, total: 0, lastDate: r.date, tipos: {} });
    g.total += r.count || 1;
    if (r.date > g.lastDate) g.lastDate = r.date;
    if (r.tipo_error) {
      g.tipos[r.tipo_error] = (g.tipos[r.tipo_error] || 0) + (r.count || 1);
    }
  }
  return Object.values(grouped);
}

/// Pantalla dedicada a mostrar los intentos que salieron "a practicar" --
/// separada de Logros (que solo cuenta lo que salió bien), para que se
/// pueda ver en qué está costando más sin mezclarlo con las estrellas.
export default function ErrorsScreen() {
  const [recordings, setRecordings] = useState(null);
  const [detail, setDetail] = useState(null);

  async function load() {
    try {
      const data = await getRecordings(getServerUrl());
      setRecordings(data.recordings || []);
    } catch {
      setRecordings((prev) => prev ?? []);
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (recordings === null) {
    return (
      <AppBackground>
        <ScreenHeader title="Para practicar" subtitle="En qué te está costando más" icon="🔁" />
        <div className="screen-loading">Cargando...</div>
      </AppBackground>
    );
  }

  const fallas = aggregate(recordings);

  return (
    <AppBackground>
      <ScreenHeader title="Para practicar" subtitle="En qué te está costando más" icon="🔁" onAction={load} />
      <div className="screen-scroll">
        {fallas.length === 0 ? (
          <EmptyState icon="🎉" message="¡Ninguna falla registrada todavía! Seguí practicando así." />
        ) : (
          <>
            <SectionLabel>Palabras a reforzar</SectionLabel>
            <div className="phrase-grid">
              {fallas.map((f, i) => (
                <FallaCard key={f.word} falla={f} color={chartPalette[i % chartPalette.length]} onOpen={() => setDetail(f)} />
              ))}
            </div>
          </>
        )}
      </div>

      {detail && <FallaDetailDialog falla={detail} onClose={() => setDetail(null)} />}
    </AppBackground>
  );
}

function FallaCard({ falla, color, onOpen }) {
  const tiposOrdenados = Object.entries(falla.tipos).sort((a, b) => b[1] - a[1]);
  return (
    <div className="phrase-card" onClick={onOpen}>
      <AppCard>
        <div className="phrase-card-top">
          <div className="phrase-card-icon" style={{ background: `${color}26` }}>
            {iconFor(falla.word)}
          </div>
          <div className="phrase-card-count" style={{ background: `${color}22`, color }}>
            {falla.total}x
          </div>
        </div>
        <div className="phrase-card-word">{falla.word.replaceAll("_", " ")}</div>
        {tiposOrdenados.length > 0 ? (
          <div className="falla-tipo-chip-list">
            {tiposOrdenados.map(([tipo, count]) => (
              <div key={tipo} className="falla-tipo-chip">
                {tipoErrorInfo(tipo).icon} {tipoErrorInfo(tipo).label} ({count}x)
              </div>
            ))}
          </div>
        ) : (
          <div className="falla-tipo-chip">🔁 Para seguir practicando ({falla.total}x)</div>
        )}
        <div className="phrase-card-spacer" />
      </AppCard>
    </div>
  );
}

function FallaDetailDialog({ falla, onClose }) {
  const tiposOrdenados = Object.entries(falla.tipos).sort((a, b) => b[1] - a[1]);
  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <div className="phrase-detail-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="phrase-detail-header">
          <div className="phrase-detail-title">
            <span className="phrase-detail-emoji">{iconFor(falla.word)}</span>
            <div className="phrase-detail-word">{falla.word.replaceAll("_", " ")}</div>
          </div>
          <button className="dismiss-button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="phrase-detail-sub">
          {falla.total} {falla.total === 1 ? "vez" : "veces"} para seguir practicando.
        </div>
        <SectionLabel>Qué pasó</SectionLabel>
        <div className="phrase-detail-people">
          {tiposOrdenados.length === 0 ? (
            <span className="phrase-detail-person-chip">Sin detalle todavía</span>
          ) : (
            tiposOrdenados.map(([tipo, count]) => {
              const info = tipoErrorInfo(tipo);
              return (
                <span key={tipo} className="phrase-detail-person-chip">
                  {info.icon} {info.label} ({count}x)
                </span>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
