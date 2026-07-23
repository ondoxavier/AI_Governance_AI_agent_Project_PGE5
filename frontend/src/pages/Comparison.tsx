import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { EmptyState, ErrorState, JurisdictionBadge, Skeleton } from "../components/States";

type Block = { status_summary?: Array<{ status: string; count: number }>; results?: Array<Record<string, unknown>>; error?: string };
export default function Comparison() {
  const [topic, setTopic] = useState("AI system used for credit decisions");
  const mutation = useMutation({ mutationFn: () => api.compare({ topic, top_k: 3 }) });
  const jurisdictions = (mutation.data?.jurisdictions ?? {}) as Record<string, Block>;
  return <div className="space-y-5"><header><h1 className="text-3xl font-semibold">Comparaison réglementaire</h1><p className="muted">Trois recherches distinctes ; aucun statut n’est fusionné.</p></header>
    <form className="panel p-4 flex flex-col md:flex-row gap-3" onSubmit={e => { e.preventDefault(); mutation.mutate(); }}><label className="grow">Sujet<input className="field mt-1" value={topic} onChange={e => setTopic(e.target.value)} /></label><button className="btn btn-primary self-end">Comparer EU / US / UK</button></form>
    {mutation.isPending && <Skeleton label="Recherche dans les trois juridictions" />}{mutation.isError && <ErrorState />}
    {mutation.isSuccess && <div className="grid lg:grid-cols-3 gap-4">{["EU", "US", "UK"].map(j => { const block = jurisdictions[j]; return <section className="panel p-4" key={j}><JurisdictionBadge value={j} />{block?.error && <ErrorState message={`Résultat partiel : ${block.error}`} />}{!block?.results?.length ? <EmptyState /> : block.results.map((item, i) => <article className="border-t border-[var(--border)] mt-4 pt-4" key={i}><strong>{String(item.title ?? "Source")}</strong><p className="muted text-sm">{String(item.date ?? "Date non renseignée")} · {String(item.status ?? "statut inconnu")}</p><p>{String(item.text ?? "")}</p><div className="mono text-xs">{String(item.source ?? "")} · score {String(item.score ?? "")}</div></article>)}</section>; })}</div>}
  </div>;
}
