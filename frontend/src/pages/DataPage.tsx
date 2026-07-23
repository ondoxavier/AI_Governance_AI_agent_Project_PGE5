import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { api } from "../api/client";
import { EmptyState, ErrorState, Skeleton } from "../components/States";

export function Observability() {
  const query = useQuery({ queryKey: ["traces"], queryFn: api.traces });
  return <DataView title="Observabilité" subtitle="Spans réellement exportés par LocalTracer ou Langfuse." query={query} />;
}
export function Evaluation() {
  const query = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluation });
  return <DataView title="Évaluation" subtitle="Baseline et résultats finaux lus depuis evaluation/latest_results.json." query={query} />;
}
function DataView({ title, subtitle, query }: { title: string; subtitle: string; query: UseQueryResult<Record<string, unknown>, Error> }) {
  if (query.isLoading) return <Skeleton />; if (query.isError) return <ErrorState />; if (!query.data || query.data.available === false) return <EmptyState />;
  return <div className="space-y-5"><header><h1 className="text-3xl font-semibold">{title}</h1><p className="muted">{subtitle}</p></header><div className="panel p-5 overflow-auto"><pre className="mono text-xs whitespace-pre-wrap">{JSON.stringify(query.data, null, 2)}</pre></div></div>;
}
