export type Jurisdiction = "EU" | "US" | "UK";
export type JurisdictionFilter = Jurisdiction | "all";
export type CriticVerdict = "APPROVE" | "REVISE";

export interface RegulatorySource {
  title: string;
  source: string;
  date: string;
  jurisdiction: Jurisdiction | string;
  status: string;
  score: number;
  method: string;
  text?: string;
}
export interface Health {
  status: string;
  agent_version: string;
  llm: { configured: boolean; provider: string };
  observability: { configured: boolean; provider: string };
  retrieval: { index_available: boolean; fallback_mode: boolean };
  mcp: { available: boolean; tools_count: number };
}
export interface JurisdictionComparisonBlock {
  jurisdiction: string;
  status: string;
  statement: string;
}
export interface AnalysisResult {
  answer: string;
  sections: {
    evidence: string[];
    analysis: string;
    conclusion: string;
    confidence: string;
    relevant_articles: string[];
    obligations: string[];
    jurisdiction_comparison: JurisdictionComparisonBlock[];
  };
  conclusion: string | null;
  confidence: number | null;
  critic_verdict: CriticVerdict;
  sources: RegulatorySource[];
  missing_information: string[];
  warnings: string[];
  trace_id: string | null;
  latency_ms: number;
  metadata: Record<string, unknown>;
  disclaimer: string;
}
export interface AnalysisJob {
  analysis_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result: AnalysisResult | null;
  error: string | null;
}
