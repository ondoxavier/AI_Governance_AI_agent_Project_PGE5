import { Activity, BarChart3, Blocks, BookOpen, GitCompare, Home, Menu, Plus, Settings, ShieldCheck, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { brand } from "../config/brand";

const navigation = [
  ["/", "Vue d’ensemble", Home],
  ["/analyses/new", "Nouvelle analyse", Plus],
  ["/comparison", "Comparaison", GitCompare],
  ["/evidence", "Preuves", BookOpen],
  ["/tools", "Outils MCP", Blocks],
  ["/observability", "Observabilité", Activity],
  ["/evaluation", "Évaluation", BarChart3],
  ["/architecture", "Architecture", ShieldCheck],
  ["/settings", "Paramètres", Settings],
] as const;

export function AppShell() {
  const [open, setOpen] = useState(false);
  return <div className="min-h-screen lg:grid lg:grid-cols-[268px_1fr]">
    <a className="skip-link" href="#main">Aller au contenu</a>
    <aside className={`${open ? "block" : "hidden"} fixed inset-0 z-40 bg-[var(--nav)] px-5 py-7 lg:static lg:block border-r border-[var(--border)]`}>
      <div className="flex justify-between items-start mb-10">
        <div><div className="eyebrow mb-2">Dossier réglementaire</div><strong className="font-serif text-2xl tracking-tight">{brand.name}</strong><div className="muted text-xs mt-1 max-w-[180px]">{brand.subtitle}</div></div>
        <button className="lg:hidden btn" onClick={() => setOpen(false)} aria-label="Fermer la navigation"><X /></button>
      </div>
      <nav aria-label="Navigation principale" className="space-y-1">
        {navigation.map(([to, label, Icon]) => <NavLink key={to} to={to} end={to === "/"} onClick={() => setOpen(false)}
          className={({ isActive }) => `flex gap-3 items-center border-l-2 px-3 py-2.5 text-sm ${isActive ? "border-[var(--accent)] bg-[var(--raised)] text-[var(--text)] font-semibold" : "border-transparent muted"}`}>
          <Icon size={18} />{label}</NavLink>)}
      </nav>
    </aside>
    <div className="min-w-0">
      <header className="h-16 px-4 md:px-8 border-b border-[var(--border)] bg-[var(--nav)] flex items-center justify-between sticky top-0 z-30">
        <button className="lg:hidden btn" onClick={() => setOpen(true)} aria-label="Ouvrir la navigation"><Menu /></button>
        <div className="hidden md:block text-xs uppercase tracking-[.12em] muted">{brand.productLine}</div>
        <NavLink className="btn btn-primary" to="/analyses/new"><Plus size={17} className="inline mr-1" />Nouvelle analyse</NavLink>
      </header>
      <main id="main" className="p-4 md:px-10 md:py-9 max-w-[1320px] mx-auto"><Outlet /></main>
    </div>
  </div>;
}
