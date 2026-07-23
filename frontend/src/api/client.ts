import { z } from "zod";
import type { AnalysisJob, Health } from "../types/api";

const healthSchema = z.object({
  status: z.string(),
  agent_version: z.string(),
  llm: z.object({ configured: z.boolean(), provider: z.string() }),
  observability: z.object({ configured: z.boolean(), provider: z.string() }),
  retrieval: z.object({ index_available: z.boolean(), fallback_mode: z.boolean() }),
  mcp: z.object({ available: z.boolean(), tools_count: z.number() }),
});

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) throw new Error(`Le service a répondu avec le statut ${response.status}.`);
  return response.json() as Promise<T>;
}

const analysisCacheKey = (id: string) => `regulaai.analysis.${id}`;

function cacheAnalysis(job: AnalysisJob): AnalysisJob {
  try {
    sessionStorage.setItem(analysisCacheKey(job.analysis_id), JSON.stringify(job));
  } catch {
    // Storage may be disabled; the API response still remains usable.
  }
  return job;
}

function cachedAnalysis(id: string): AnalysisJob | null {
  try {
    const value = sessionStorage.getItem(analysisCacheKey(id));
    return value ? JSON.parse(value) as AnalysisJob : null;
  } catch {
    return null;
  }
}

export const api = {
  health: async (): Promise<Health> => healthSchema.parse(await request("/health")),
  corpus: () => request<Record<string, unknown>>("/corpus"),
  tools: () => request<{ tools: Array<Record<string, unknown>> }>("/tools"),
  evaluation: () => request<Record<string, unknown>>("/evaluation/latest"),
  traces: () => request<Record<string, unknown>>("/traces/latest"),
  search: (body: unknown) => request<Record<string, unknown>>("/search", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: unknown) => request<Record<string, unknown>>("/compare", { method: "POST", body: JSON.stringify(body) }),
  invoke: (name: string, arguments_: unknown) => request<Record<string, unknown>>(`/tools/${name}/invoke`, { method: "POST", body: JSON.stringify({ arguments: arguments_ }) }),
  createAnalysis: async (body: unknown) => cacheAnalysis(await request<AnalysisJob>("/analyses", { method: "POST", body: JSON.stringify(body) })),
  analysis: async (id: string) => cachedAnalysis(id) ?? cacheAnalysis(await request<AnalysisJob>(`/analyses/${id}`)),
};
