import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { ErrorState, Skeleton } from "../components/States";

export default function Tools() {
  const tools = useQuery({ queryKey: ["tools"], queryFn: api.tools }); const [selected, setSelected] = useState("hybrid_search"); const [input, setInput] = useState("AI credit scoring");
  const invoke = useMutation({ mutationFn: () => api.invoke(selected, selected === "security_screen" ? { text: input } : selected === "compare_jurisdiction" ? { topic: input, top_k: 3 } : { query: input, ...(selected === "hybrid_search" ? { top_k: 3, jurisdiction: "all" } : {}) }) });
  if (tools.isLoading) return <Skeleton />; if (tools.error || !tools.data) return <ErrorState />;
  return <div className="space-y-5"><header><h1 className="text-3xl font-semibold">Outils MCP</h1><p className="muted">Liste issue directement de MCP_TOOL_REGISTRY.</p></header>
    <div className="grid lg:grid-cols-[320px_1fr] gap-4"><div className="panel p-3">{tools.data.tools.map(tool => <button className={`block text-left w-full p-3 rounded-lg ${selected === tool.name ? "bg-[var(--raised)]" : ""}`} key={String(tool.name)} onClick={() => setSelected(String(tool.name))}><strong className="mono">{String(tool.name)}</strong><div className="muted text-xs mt-1">Risque surveillé</div></button>)}</div>
    <section className="panel p-5"><h2 className="text-xl mono">{selected}</h2><form className="mt-4 space-y-3" onSubmit={e => { e.preventDefault(); invoke.mutate(); }}><label>Requête ou texte<input className="field mt-1" value={input} onChange={e => setInput(e.target.value)} /></label><button className="btn btn-primary">Tester l’outil réel</button></form>{invoke.isError && <ErrorState />}{invoke.data && <pre tabIndex={0} className="mono text-xs bg-[var(--canvas)] p-4 rounded-lg overflow-auto mt-4 max-h-[500px]">{JSON.stringify(invoke.data, null, 2)}</pre>}</section></div>
  </div>;
}
