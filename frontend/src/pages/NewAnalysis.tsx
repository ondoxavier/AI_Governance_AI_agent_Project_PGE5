import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import { api } from "../api/client";

const schema = z.object({
  question: z.string().min(10, "Décrivez le système en au moins 10 caractères.").max(4000),
  jurisdiction: z.enum(["all", "EU", "US", "UK"]),
  top_k: z.coerce.number().min(1).max(10),
  self_consistency_k: z.coerce.number().min(3).max(5),
});
type Form = z.infer<typeof schema>;

const banking = "Une banque opérant en Europe et aux États-Unis utilise un modèle d’IA pour présélectionner des candidats à un crédit. Identifier le niveau de risque AI Act, les obligations et les cadres US et UK applicables.";

export default function NewAnalysis() {
  const navigate = useNavigate(); const [submitting, setSubmitting] = useState(false);
  const { register, handleSubmit, setValue, formState: { errors } } = useForm<Form>({ resolver: zodResolver(schema), defaultValues: { question: banking, jurisdiction: "all", top_k: 5, self_consistency_k: 3 } });
  const submit = async (data: Form) => { setSubmitting(true); try { const job = await api.createAnalysis({ ...data, mode: "free" }); navigate(`/analyses/${job.analysis_id}`); } finally { setSubmitting(false); } };
  return <div className="max-w-4xl space-y-5">
    <header><h1 className="text-3xl font-semibold">Nouvelle analyse</h1><p className="muted">Le texte passe par le garde-fou L1 puis par le véritable pipeline de l’agent.</p></header>
    <form className="panel p-5 space-y-5" onSubmit={handleSubmit(submit)} noValidate>
      <div><label htmlFor="question" className="block font-medium mb-2">Description ou question réglementaire</label><textarea id="question" rows={9} className="field" aria-invalid={!!errors.question} aria-describedby="question-error" {...register("question")} />{errors.question && <p id="question-error" role="alert" className="text-[var(--critical)]">{errors.question.message}</p>}</div>
      <button type="button" className="btn" onClick={() => setValue("question", banking)}>Remplir le scénario bancaire</button>
      <div className="grid md:grid-cols-3 gap-4">
        <label>Juridiction<select className="field mt-1" {...register("jurisdiction")}><option value="all">Toutes</option><option>EU</option><option>US</option><option>UK</option></select></label>
        <label>Top-k<input className="field mt-1" type="number" {...register("top_k")} /></label>
        <label>Self-consistency<input className="field mt-1" type="number" {...register("self_consistency_k")} /></label>
      </div>
      <button className="btn btn-primary" disabled={submitting}>{submitting ? "Mise en file…" : "Lancer l’analyse"}</button>
    </form>
  </div>;
}
