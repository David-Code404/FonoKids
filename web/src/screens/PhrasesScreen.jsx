import { useEffect, useMemo, useState } from "react";
import { getRecordings, getServerUrl } from "../api.js";
import { AppCard, AppBackground, EmptyState, ErrorState, ScreenHeader, SectionLabel } from "../components/Shared.jsx";
import { chartPalette } from "../theme.js";
import "./Screens.css";

const SORT_MODES = [
  { key: "mostSaid", icon: "🔥", label: "Más dichas primero" },
  { key: "recent", icon: "🕓", label: "Más recientes primero" },
  { key: "alphabetical", icon: "🔤", label: "Alfabético" },
];

function hashCode(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function aggregate(recordings) {
  const grouped = {};
  for (const r of recordings) {
    (grouped[r.word] ??= []).push(r);
  }
  return Object.entries(grouped).map(([word, items]) => {
    const sorted = [...items].sort((a, b) => b.date.localeCompare(a.date));
    return {
      word,
      count: sorted.length,
      lastDate: sorted[0].date,
      people: [...new Set(sorted.map((r) => r.person))],
    };
  });
}

/// Frases de riesgo agrupadas (cuántas veces se dijo cada una, quién y
/// cuándo) -- igual que phrases_screen.dart (PhrasesScreen).
export default function PhrasesScreen() {
  const [recordings, setRecordings] = useState(null);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState("mostSaid");
  const [sortMenuOpen, setSortMenuOpen] = useState(false);
  const [detail, setDetail] = useState(null);

  async function load() {
    setError(null);
    try {
      const serverUrl = getServerUrl();
      const data = await getRecordings(serverUrl);
      setRecordings(data.recordings || []);
    } catch (e) {
      // Ver el mismo comentario en CapturesScreen.jsx -- no pisar datos
      // reales ya cargados en un refresh fallido, y nunca caer a datos
      // inventados: si no hay nada real todavía, se muestra el error.
      setRecordings((prev) => {
        if (!prev || prev.length === 0) {
          setError(e.message || "No se pudo conectar al servidor.");
          return [];
        }
        return prev;
      });
    }
  }

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    if (!recordings) return [];
    const all = aggregate(recordings);
    const sorted = [...all].sort((a, b) => {
      if (sortMode === "alphabetical") return a.word.localeCompare(b.word);
      if (sortMode === "recent") return b.lastDate.localeCompare(a.lastDate);
      return b.count - a.count;
    });
    if (!query) return sorted;
    return sorted.filter((w) => w.word.replaceAll("_", " ").toLowerCase().includes(query));
  }, [recordings, sortMode, query]);

  return (
    <AppBackground>
      <ScreenHeader title="Frases" subtitle="Frases de riesgo detectadas" icon="🗣" />

      <div className="phrases-toolbar">
        <div className="search-field">
          <span className="search-icon">🔍</span>
          <input
            placeholder="Buscar frase..."
            onChange={(e) => setQuery(e.target.value.trim().toLowerCase())}
          />
        </div>
        <div className="sort-button-wrap">
          <button className="sort-button" onClick={() => setSortMenuOpen((v) => !v)}>
            ⇅
          </button>
          {sortMenuOpen && (
            <div className="sort-menu">
              {SORT_MODES.map((m) => (
                <button
                  key={m.key}
                  className={`sort-menu-item ${sortMode === m.key ? "selected" : ""}`}
                  onClick={() => {
                    setSortMode(m.key);
                    setSortMenuOpen(false);
                  }}
                >
                  {m.icon} {m.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="screen-scroll">
        {recordings === null && !error && <div className="screen-loading">Cargando...</div>}
        {error && <ErrorState message={error} onRetry={load} />}
        {recordings !== null && !error && aggregate(recordings).length === 0 && (
          <EmptyState icon="🗣" message="Todavía no se detectó ninguna frase de riesgo." />
        )}
        {recordings !== null && !error && aggregate(recordings).length > 0 && filtered.length === 0 && (
          <EmptyState icon="🔍✕" message="Ninguna frase coincide con la búsqueda." />
        )}
        {filtered.length > 0 && (
          <div className="phrase-grid">
            {filtered.map((agg, i) => (
              <PhraseCard
                key={agg.word}
                agg={agg}
                color={chartPalette[i % chartPalette.length]}
                onOpen={() => setDetail(agg)}
              />
            ))}
          </div>
        )}
      </div>

      {detail && <PhraseDetailDialog agg={detail} onClose={() => setDetail(null)} />}
    </AppBackground>
  );
}

function PhraseCard({ agg, color, onOpen }) {
  return (
    <div className="phrase-card" onClick={onOpen}>
      <AppCard>
        <div className="phrase-card-top">
          <div className="phrase-card-icon" style={{ background: `${color}26`, color }}>
            🗣
          </div>
          <div className="phrase-card-count">{agg.count}x</div>
        </div>
        <div className="phrase-card-word">{agg.word.replaceAll("_", " ")}</div>
        <div className="phrase-card-spacer" />
        <div className="phrase-card-last">🕓 Última vez: {agg.lastDate}</div>
        <PeopleRow people={agg.people} />
      </AppCard>
    </div>
  );
}

function PeopleRow({ people }) {
  const shown = people.slice(0, 3);
  const extra = people.length - shown.length;
  return (
    <div className="people-row">
      <div className="people-stack" style={{ width: shown.length * 14 + 6 }}>
        {shown.map((p, i) => (
          <div
            key={p}
            className="people-avatar"
            style={{ left: i * 14, background: chartPalette[hashCode(p) % chartPalette.length] }}
          >
            {p.trim().slice(0, 1).toUpperCase() || "?"}
          </div>
        ))}
      </div>
      {extra > 0 && <span className="people-extra">+{extra}</span>}
    </div>
  );
}

function PhraseDetailDialog({ agg, onClose }) {
  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <div className="phrase-detail-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="phrase-detail-header">
          <div className="phrase-detail-word">{agg.word.replaceAll("_", " ")}</div>
          <button className="dismiss-button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="phrase-detail-sub">
          Se dijo {agg.count} {agg.count === 1 ? "vez" : "veces"} -- última el {agg.lastDate}.
        </div>
        <SectionLabel>Personas</SectionLabel>
        <div className="phrase-detail-people">
          {agg.people.map((p) => (
            <span key={p} className="phrase-detail-person-chip">
              {p.replaceAll("-", " ")}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
