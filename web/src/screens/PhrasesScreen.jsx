import { useEffect, useMemo, useState } from "react";
import { getRecordings, getServerUrl } from "../api.js";
import { mockRecordings } from "../mockData.js";
import { iconFor } from "../wordIcons.js";
import { AppCard, AppBackground, EmptyState, ScreenHeader, SectionLabel } from "../components/Shared.jsx";
import { chartPalette } from "../theme.js";
import "./Screens.css";

const SORT_MODES = [
  { key: "mostSaid", icon: "🔥", label: "Más dichas primero" },
  { key: "recent", icon: "🕓", label: "Más recientes primero" },
  { key: "alphabetical", icon: "🔤", label: "Alfabético" },
];

function starsFor(pct) {
  if (pct >= 80) return 3;
  if (pct >= 50) return 2;
  if (pct > 0) return 1;
  return 0;
}

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
    const correctCount = items.filter((r) => r.correcta).reduce((sum, r) => sum + r.count, 0);
    const incorrectCount = items.filter((r) => !r.correcta).reduce((sum, r) => sum + r.count, 0);
    return {
      word,
      count: correctCount + incorrectCount,
      correctCount,
      incorrectCount,
      lastDate: sorted[0].date,
      people: [...new Set(sorted.map((r) => r.person))],
    };
  });
}

/// Palabras objetivo agrupadas (progreso de pronunciación por palabra: cuántas
/// veces se dijo bien/mal, quién y cuándo) -- igual que phrases_screen.dart
/// (PhrasesScreen), reconvertido a seguimiento de pronunciación.
export default function PhrasesScreen() {
  const [recordings, setRecordings] = useState(null);
  const [usingMock, setUsingMock] = useState(false);
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState("mostSaid");
  const [sortMenuOpen, setSortMenuOpen] = useState(false);
  const [detail, setDetail] = useState(null);

  async function load() {
    try {
      const serverUrl = getServerUrl();
      const data = await getRecordings(serverUrl);
      const real = data.recordings || [];
      // Ver el mismo comentario en CapturesScreen.jsx -- mientras no haya
      // dataset real, se muestran datos de EJEMPLO marcados como tales.
      setUsingMock(real.length === 0);
      setRecordings(real.length > 0 ? real : mockRecordings());
    } catch (e) {
      setRecordings((prev) => {
        if (!prev || prev.length === 0 || usingMock) {
          setUsingMock(true);
          return mockRecordings();
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
      <ScreenHeader title="Logros" subtitle="Tus estrellas por palabra" icon="🏆" />

      <div className="phrases-toolbar">
        <div className="search-field">
          <span className="search-icon">🔍</span>
          <input
            placeholder="Buscar palabra..."
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
        {usingMock && recordings !== null && (
          <div className="mock-banner">📋 Mostrando datos de ejemplo -- todavía no hay práctica real guardada.</div>
        )}
        {recordings === null && <div className="screen-loading">Cargando...</div>}
        {recordings !== null && aggregate(recordings).length === 0 && (
          <EmptyState icon="🗣" message="Todavía no se practicó ninguna palabra." />
        )}
        {recordings !== null && aggregate(recordings).length > 0 && filtered.length === 0 && (
          <EmptyState icon="🔍✕" message="Ninguna palabra coincide con la búsqueda." />
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
  const pct = agg.count > 0 ? Math.round((100 * agg.correctCount) / agg.count) : 0;
  const stars = starsFor(pct);
  return (
    <div className="phrase-card" onClick={onOpen}>
      <AppCard>
        <div className="phrase-card-top">
          <div className="phrase-card-icon" style={{ background: `${color}26` }}>
            {iconFor(agg.word)}
          </div>
          <div className="phrase-card-count" style={{ background: `${color}22`, color }}>
            {agg.count}x
          </div>
        </div>
        <div className="phrase-card-word">{agg.word.replaceAll("_", " ")}</div>
        <div className="phrase-card-stars">
          {[1, 2, 3].map((n) => (
            <span key={n} className={n <= stars ? "star filled" : "star"}>
              ⭐
            </span>
          ))}
        </div>
        <div className="phrase-card-bar-track">
          <div className="phrase-card-bar-fill" style={{ width: `${pct}%`, background: color }} />
        </div>
        <div className="phrase-card-progress">
          ✅ {agg.correctCount} · 🔁 {agg.incorrectCount}
        </div>
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
          <div className="phrase-detail-title">
            <span className="phrase-detail-emoji">{iconFor(agg.word)}</span>
            <div className="phrase-detail-word">{agg.word.replaceAll("_", " ")}</div>
          </div>
          <button className="dismiss-button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="phrase-detail-sub">
          Practicada {agg.count} {agg.count === 1 ? "vez" : "veces"} -- {agg.correctCount} bien dichas,{" "}
          {agg.incorrectCount} para seguir practicando -- última el {agg.lastDate}.
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
