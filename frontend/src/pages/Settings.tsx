import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { ErrorState, Skeleton } from "../components/States";
export default function Settings() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health }); const [theme, setTheme] = useState(localStorage.getItem("regula-theme") ?? "light");
  if (health.isLoading) return <Skeleton />; if (!health.data) return <ErrorState />;
  const update = (value: string) => { setTheme(value); localStorage.setItem("regula-theme", value); document.documentElement.dataset.theme = value; };
  return <div className="space-y-5"><header><h1 className="text-3xl font-semibold">Paramètres non sensibles</h1><p className="muted">Aucune clé n’est lisible ou modifiable depuis le navigateur.</p></header><section className="panel p-5 grid-cards"><div>LLM configuré : <strong>{health.data.llm.configured ? "oui" : "non"}</strong></div><div>Langfuse configuré : <strong>{health.data.observability.configured ? "oui" : "non"}</strong></div><div>Index disponible : <strong>{health.data.retrieval.index_available ? "oui" : "non"}</strong></div><div>Version : <strong className="mono">{health.data.agent_version}</strong></div></section><section className="panel p-5"><label>Thème<select className="field mt-2 max-w-xs" value={theme} onChange={e => update(e.target.value)}><option value="dark">Sombre</option><option value="light">Clair</option></select></label></section></div>;
}
