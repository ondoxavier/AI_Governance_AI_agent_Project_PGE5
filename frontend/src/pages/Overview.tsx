import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { brand } from "../config/brand";
import { ErrorState, Skeleton } from "../components/States";

export default function Overview() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const corpus = useQuery({ queryKey: ["corpus"], queryFn: api.corpus });
  if (health.isLoading) return <Skeleton label="Vérification du système" />;
  if (health.error || !health.data) return <ErrorState />;
  const h = health.data;
  return <div className="space-y-9">
    <header className="max-w-4xl border-b border-[var(--border)] pb-8"><p className="eyebrow mb-3">Note de synthèse</p><h1 className="text-4xl md:text-5xl font-normal">{brand.name}</h1><p className="muted text-lg mt-3 max-w-2xl">{brand.productLine}</p></header>
    <div className="grid lg:grid-cols-[1.5fr_1fr] gap-7 items-start">
    <section className="panel p-6 md:p-8"><p className="eyebrow mb-3">Étude en cours</p><h2 className="text-2xl font-normal">Scénario bancaire transatlantique</h2><p className="muted max-w-2xl mt-3">Évaluer un modèle bancaire qui présélectionne des candidats au crédit en Europe et aux États-Unis, puis distinguer les obligations applicables dans chaque juridiction.</p><Link className="btn btn-primary inline-block mt-6" to="/analyses/new">Ouvrir le dossier d’analyse</Link></section>
    <section className="panel p-5 md:p-6" aria-label="État du système">
      <h2 className="text-lg font-normal mb-3">Disponibilité des services</h2>
      {[
        ["Service d’analyse", h.status === "ok" ? "Disponible" : "Indisponible"],
        ["Moteur de raisonnement", h.llm.configured ? `${h.llm.provider} configuré` : "Mode local"],
        ["Base documentaire", h.retrieval.index_available ? "Index disponible" : "Corpus essentiel"],
        ["Traçabilité", h.observability.configured ? "Langfuse configuré" : "Journal local"],
      ].map(([label, value]) => <div className="status-row" key={label}><div className="muted text-sm">{label}</div><strong className="text-sm"><span className="status-dot" />{value}</strong></div>)}
    </section>
    </div>
    {h.retrieval.fallback_mode && <div className="border-l-2 border-[var(--warn)] pl-4 py-1 text-sm muted" role="status">Le corpus essentiel est actuellement utilisé. Les conclusions restent soumises à validation juridique.</div>}
    <details className="border-t border-[var(--border)] pt-4"><summary className="text-sm font-semibold cursor-pointer">Informations techniques sur le corpus</summary><pre className="mono text-xs overflow-auto mt-3 panel p-4">{corpus.isSuccess ? JSON.stringify(corpus.data, null, 2) : "Chargement…"}</pre></details>
  </div>;
}
