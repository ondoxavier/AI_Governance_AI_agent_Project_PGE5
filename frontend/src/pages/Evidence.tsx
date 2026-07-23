import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { EmptyState, ErrorState, JurisdictionBadge, Skeleton } from "../components/States";

export default function Evidence() {
  const [query, setQuery] = useState("AI credit scoring"); const [jurisdiction, setJurisdiction] = useState("EU"); const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const search = useMutation({ mutationFn: () => api.search({ query, jurisdiction, top_k: 5 }) });
  const results = (search.data?.results ?? []) as Array<Record<string, unknown>>;
  return <div className="space-y-5"><header><h1 className="text-3xl font-semibold">Explorateur de preuves</h1></header>
    <form className="panel p-4 grid md:grid-cols-[1fr_140px_auto] gap-3" onSubmit={e => { e.preventDefault(); search.mutate(); }}><label>Recherche<input className="field mt-1" value={query} onChange={e => setQuery(e.target.value)} /></label><label>Juridiction<select className="field mt-1" value={jurisdiction} onChange={e => setJurisdiction(e.target.value)}><option>EU</option><option>US</option><option>UK</option><option value="all">Toutes</option></select></label><button className="btn btn-primary self-end">Rechercher</button></form>
    {search.isPending && <Skeleton />}{search.isError && <ErrorState />}
    {search.isSuccess && !results.length && <EmptyState />}
    {!!results.length && <div className="grid lg:grid-cols-2 gap-4"><div className="panel p-3">{results.map((item, i) => <button key={i} onClick={() => setSelected(item)} className="text-left block w-full p-4 border-b border-[var(--border)] hover:bg-[var(--raised)]"><JurisdictionBadge value={String(item.jurisdiction)} /><strong className="block mt-2">{String(item.title)}</strong><span className="muted text-sm">{String(item.status)} · score {String(item.score)}</span></button>)}</div><article className="panel p-5">{selected ? <><h2 className="text-xl font-semibold">{String(selected.title)}</h2><p className="muted">{String(selected.date)} · {String(selected.status)}</p><p className="my-4 whitespace-pre-wrap">{String(selected.text)}</p><div className="mono text-xs break-all">{String(selected.source)}</div><a className="btn btn-primary inline-block mt-4" href={`/analyses/new?seed=${encodeURIComponent(String(selected.text))}`}>Utiliser comme point de départ</a></> : <EmptyState message="Sélectionnez une preuve pour consulter son détail." />}</article></div>}
  </div>;
}
