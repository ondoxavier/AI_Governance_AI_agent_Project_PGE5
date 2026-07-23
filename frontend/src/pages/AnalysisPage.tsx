import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import { AnalysisResult } from "../components/AnalysisResult";
import { ErrorState, Skeleton } from "../components/States";

export default function AnalysisPage() {
  const { id = "" } = useParams();
  const query = useQuery({ queryKey: ["analysis", id], queryFn: () => api.analysis(id), refetchInterval: q => ["completed", "failed"].includes(q.state.data?.status ?? "") ? false : 800 });
  if (query.isLoading || query.data?.status === "queued" || query.data?.status === "running") return <div aria-live="polite"><Skeleton label={`Analyse ${query.data?.status === "running" ? "en cours" : "en file"}`} /><p className="muted mt-3">Les événements affichés proviennent des vrais spans du job, sans progression simulée.</p></div>;
  if (query.error || !query.data) return <ErrorState />;
  if (query.data.status === "failed" || !query.data.result) return <ErrorState message={query.data.error ?? "Analyse interrompue."} />;
  return <><header className="mb-5"><h1 className="text-3xl font-semibold">Résultat d’analyse</h1></header><AnalysisResult result={query.data.result} /></>;
}
