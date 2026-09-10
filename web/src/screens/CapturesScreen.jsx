import { useEffect, useState } from "react";
import { getRecordings, getServerUrl, thumbnailUrl } from "../api.js";
import { mockRecordings } from "../mockData.js";
import { AppBackground, AppCard, EmptyState, ErrorState, HeroCard, ScreenHeader, SectionLabel } from "../components/Shared.jsx";
import { chartPalette } from "../theme.js";
import "./Screens.css";

const AVATAR_PALETTE = [chartPalette[0], chartPalette[1], chartPalette[2], "#FF6B81", "#5AA9FF"];

function hashCode(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function initialsFor(person) {
  const name = person.replaceAll("-", " ").trim();
  if (!name || name === "desconocido") return "?";
  const parts = name.split(" ").filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 1).toUpperCase();
  return (parts[0].slice(0, 1) + parts[parts.length - 1].slice(0, 1)).toUpperCase();
}

/// Historial de sesiones grabadas, agrupadas por fecha -- igual que
/// captures_screen.dart (CapturesScreen).
export default function CapturesScreen() {
  const [recordings, setRecordings] = useState(null);
  const [error, setError] = useState(null);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [dialogRecording, setDialogRecording] = useState(null);

  async function load() {
    setError(null);
    try {
      const serverUrl = getServerUrl();
      const data = await getRecordings(serverUrl);
      setRecordings(data.recordings || []);
    } catch (e) {
      // Si YA había datos reales cargados, un refresh fallido (ej. el
      // servidor está ocupado con una predicción) NO debe reemplazarlos por
      // datos de muestra -- eso hacía que la lista pareciera "cambiar sola".
      // Los de muestra son solo para la primerísima carga, sin servidor.
      setRecordings((prev) => (prev && prev.length > 0 ? prev : mockRecordings()));
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (recordings === null) {
    return (
      <AppBackground>
        <ScreenHeader title="Capturas" subtitle="Historial de sesiones grabadas" icon="🗓" />
        <div className="screen-loading">Cargando...</div>
      </AppBackground>
    );
  }

  if (error) {
    return (
      <AppBackground>
        <ScreenHeader title="Capturas" subtitle="Historial de sesiones grabadas" icon="🗓" />
        <ErrorState message={error} onRetry={load} />
      </AppBackground>
    );
  }

  if (recordings.length === 0) {
    return (
      <AppBackground>
        <ScreenHeader title="Capturas" subtitle="Historial de sesiones grabadas" icon="🗓" />
        <EmptyState icon="🗓✕" message="Todavía no hay grabaciones." />
      </AppBackground>
    );
  }

  const people = [...new Set(recordings.map((r) => r.person))].sort();
  const visible = selectedPerson ? recordings.filter((r) => r.person === selectedPerson) : recordings;

  const grouped = {};
  for (const r of visible) {
    (grouped[r.date] ??= []).push(r);
  }
  // La más nueva siempre arriba -- "HH:MM" ordena bien como texto, y las
  // que no tienen hora (mock/datos viejos sin ese campo) quedan al final.
  for (const items of Object.values(grouped)) {
    items.sort((a, b) => (b.time || "").localeCompare(a.time || ""));
  }
  const dates = Object.keys(grouped).sort((a, b) => b.localeCompare(a));

  return (
    <AppBackground>
      <ScreenHeader title="Capturas" subtitle="Historial de sesiones grabadas" icon="🗓" onAction={load} />
      <div className="screen-scroll">
        <HeroCard>
          <div className="summary-strip">
            <div className="summary-stat">
              <div className="summary-icon">🎥</div>
              <div className="summary-value">{recordings.length}</div>
              <div className="summary-label">Sesiones</div>
            </div>
            <div className="summary-divider" />
            <div className="summary-stat">
              <div className="summary-icon">📅</div>
              <div className="summary-value">{new Set(recordings.map((r) => r.date)).size}</div>
              <div className="summary-label">Días activos</div>
            </div>
            <div className="summary-divider" />
            <div className="summary-stat">
              <div className="summary-icon">👥</div>
              <div className="summary-value">{people.length}</div>
              <div className="summary-label">Personas</div>
            </div>
          </div>
        </HeroCard>

        <SectionLabel>Filtrar por persona</SectionLabel>
        <div className="person-filter-row">
          <button
            className={`filter-chip ${selectedPerson === null ? "selected" : ""}`}
            onClick={() => setSelectedPerson(null)}
          >
            Todos
          </button>
          {people.map((p) => (
            <button
              key={p}
              className={`filter-chip ${selectedPerson === p ? "selected" : ""}`}
              onClick={() => setSelectedPerson(p)}
            >
              {p.replaceAll("-", " ")}
            </button>
          ))}
        </div>

        <SectionLabel>Línea de tiempo</SectionLabel>
        <div className="timeline">
          {dates.length === 0 ? (
            <EmptyState icon="🔍" message="Esta persona todavía no grabó ninguna frase." />
          ) : (
            dates.map((date, i) => (
              <TimelineDay key={date} date={date} items={grouped[date]} isLast={i === dates.length - 1}
                onOpen={setDialogRecording} />
            ))
          )}
        </div>
      </div>

      {dialogRecording && (
        <CaptureDialog recording={dialogRecording} onClose={() => setDialogRecording(null)} />
      )}
    </AppBackground>
  );
}

function TimelineDay({ date, items, isLast, onOpen }) {
  return (
    <div className="timeline-day">
      <div className="timeline-rail">
        <div className="timeline-dot" />
        {!isLast && <div className="timeline-line" />}
      </div>
      <div className="timeline-content">
        <div className="timeline-date-header">
          <span className="timeline-date">{date}</span>
          <span className="timeline-count">
            · {items.length} {items.length === 1 ? "frase" : "frases"}
          </span>
        </div>
        <AppCard padded={false}>
          {items.map((r, j) => (
            <RecordingTile key={j} recording={r} isLast={j === items.length - 1} onOpen={onOpen} />
          ))}
        </AppCard>
      </div>
    </div>
  );
}

function RecordingTile({ recording, isLast, onOpen }) {
  const color = AVATAR_PALETTE[hashCode(recording.person) % AVATAR_PALETTE.length];
  return (
    <div className={`recording-tile ${isLast ? "" : "with-border"}`}>
      <div className="recording-avatar" style={{ background: `${color}2e`, color }}>
        {initialsFor(recording.person)}
      </div>
      <div className="recording-info">
        <div className="recording-word">{recording.word.replaceAll("_", " ")}</div>
        <div className="recording-person">
          {recording.person.replaceAll("-", " ")}
          {recording.time && <span className="recording-time"> · {recording.time}</span>}
        </div>
      </div>
      <button className="recording-camera-btn" style={{ color }} onClick={() => onOpen(recording)} title="Ver captura">
        📷
      </button>
    </div>
  );
}

function CaptureDialog({ recording, onClose }) {
  const color = AVATAR_PALETTE[hashCode(recording.person) % AVATAR_PALETTE.length];
  const initials = initialsFor(recording.person);
  const personName = recording.person.replaceAll("-", " ").trim();
  const displayName = !personName || personName === "desconocido" ? "Persona no identificada" : personName;

  const [photoFailed, setPhotoFailed] = useState(false);
  const photoUrl =
    recording.thumbnail_file && !photoFailed
      ? thumbnailUrl(getServerUrl(), recording.word, recording.thumbnail_file)
      : null;

  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <div className="capture-dialog" onClick={(e) => e.stopPropagation()}>
        <div
          className="capture-dialog-image"
          style={photoUrl ? undefined : { background: `linear-gradient(135deg, ${color}8c, ${color}2e)` }}
        >
          {photoUrl ? (
            <img
              className="capture-dialog-photo"
              src={photoUrl}
              alt={`Foto de la captura: ${recording.word}`}
              onError={() => setPhotoFailed(true)}
            />
          ) : (
            <>
              <div className="capture-dialog-avatar" style={{ borderColor: "rgba(255,255,255,0.4)" }}>
                {initials}
              </div>
              <div className="capture-dialog-tag">Imagen de muestra -- sin foto real todavía</div>
            </>
          )}
          <button className="capture-dialog-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="capture-dialog-body">
          <div className="capture-dialog-title">{recording.word.replaceAll("_", " ")}</div>
          <div className="capture-dialog-row">👤 {displayName}</div>
          <div className="capture-dialog-row">
            📅 {recording.date}
            {recording.time && ` · ${recording.time}`}
          </div>
        </div>
      </div>
    </div>
  );
}
