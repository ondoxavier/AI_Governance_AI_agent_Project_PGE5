import { Check, Copy, Download, Printer, TriangleAlert } from "lucide-react";
import type { AnalysisResult as Result } from "../types/api";
import { HumanValidationBanner, JurisdictionBadge } from "./States";

export function AnalysisResult({ result }: { result: Result }) {
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `regulaai-${result.trace_id ?? "analyse"}.json`; link.click(); URL.revokeObjectURL(url);
  };
  return <div className="space-y-5">
    <div className="flex flex-wrap gap-2">
      <span className="badge">{result.critic_verdict === "APPROVE" ? <Check size={14} /> : <TriangleAlert size={14} />} Verdict {result.critic_verdict}</span>
      <span className="badge mono">Confiance {result.confidence == null ? "non disponible" : `${Math.round(result.confidence * 100)} %`}</span>
      <span className="badge mono">{Math.round(result.latency_ms)} ms</span>
      <button className="btn" onClick={() => navigator.clipboard.writeText(result.conclusion ?? "")}><Copy size={15} className="inline" /> Conclusion</button>
      <button className="btn" onClick={download}><Download size={15} className="inline" /> JSON</button>
      <button className="btn" onClick={() => window.print()}><Printer size={15} className="inline" /> Imprimer</button>
    </div>
    <section className="panel p-5"><h2 className="text-lg font-semibold">Conclusion</h2><p>{result.conclusion ?? "Conclusion indisponible."}</p></section>
    <section className="panel p-5"><h2 className="text-lg font-semibold">Analyse</h2><p className="whitespace-pre-wrap">{result.sections.analysis || result.answer}</p></section>
    {result.missing_information.length > 0 && <section className="panel p-5 border-[var(--warn)]"><h2 className="font-semibold">Informations manquantes</h2><ul className="list-disc pl-5">{result.missing_information.map(x => <li key={x}>{x}</li>)}</ul></section>}
    {result.warnings.length > 0 && <section className="panel p-5"><h2 className="font-semibold">Avertissements</h2><ul className="list-disc pl-5">{result.warnings.map(x => <li key={x}>{x}</li>)}</ul></section>}
    <section><h2 className="text-lg font-semibold mb-3">Sources</h2><div className="source-grid">{result.sources.map((source, index) => <article className="panel p-4 min-w-0" key={`${source.source}-${index}`}>
      <div className="flex justify-between gap-2"><JurisdictionBadge value={source.jurisdiction} /><span className="badge">{source.status}</span></div>
      <h3 className="font-semibold my-3">{source.title}</h3><div className="muted text-sm">{source.date}</div>
      <div className="mono text-xs mt-3">score {source.score} · {source.method}</div><div className="mono text-xs break-words mt-2">{source.source}</div>
    </article>)}</div></section>
    <div className="mono muted text-xs">Trace : {result.trace_id ?? "non disponible"}</div>
    <HumanValidationBanner />
  </div>;
}
