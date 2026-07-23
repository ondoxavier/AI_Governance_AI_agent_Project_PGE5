import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { AnalysisResult } from "./AnalysisResult";
test("affiche les éléments réglementaires essentiels", () => {
  render(<AnalysisResult result={{answer:"Réponse",sections:{evidence:[],analysis:"Analyse",conclusion:"Conclusion",confidence:"0.8"},conclusion:"Conclusion",confidence:.8,critic_verdict:"REVISE",sources:[{title:"AI Act",source:"act.pdf",date:"2024",jurisdiction:"EU",status:"obligatoire",score:.7,method:"hybrid"}],missing_information:["autonomie"],warnings:[],trace_id:"trace-1",latency_ms:100,metadata:{},disclaimer:"Validation"}} />);
  expect(screen.getByText(/Verdict REVISE/)).toBeInTheDocument();
  expect(screen.getByText(/Confiance 80/)).toBeInTheDocument();
  expect(screen.getByText("AI Act")).toBeInTheDocument();
  expect(screen.getByText(/validée par un juriste/)).toBeInTheDocument();
});
