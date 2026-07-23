import { AlertTriangle, Database, LoaderCircle, WifiOff } from "lucide-react";

export function Skeleton({ label = "Chargement" }: { label?: string }) {
  return <div className="panel p-6 muted" role="status"><LoaderCircle className="inline animate-spin mr-2" size={18} />{label}…</div>;
}
export function ErrorState({ message = "Le service est momentanément indisponible." }: { message?: string }) {
  return <div className="panel p-6" role="alert"><AlertTriangle className="inline mr-2" color="var(--critical)" />{message}</div>;
}
export function EmptyState({ message = "Aucune donnée disponible." }: { message?: string }) {
  return <div className="panel p-6 muted"><Database className="inline mr-2" />{message}</div>;
}
export function OfflineState() {
  return <div className="panel p-6" role="alert"><WifiOff className="inline mr-2" />Connexion au backend indisponible.</div>;
}
export function JurisdictionBadge({ value }: { value: string }) {
  return <span className="badge" aria-label={`Juridiction ${value}`}>{value === "EU" ? "●" : value === "US" ? "◆" : "■"} {value}</span>;
}
export function HumanValidationBanner() {
  return <div className="disclaimer" role="note">⚠ Cette analyse est générée par IA et doit être validée par un juriste avant toute décision.</div>;
}
