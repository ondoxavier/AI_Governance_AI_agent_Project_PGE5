import { lazy, Suspense } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "../layouts/AppShell";
import { Skeleton } from "../components/States";
const Overview = lazy(() => import("../pages/Overview"));
const NewAnalysis = lazy(() => import("../pages/NewAnalysis"));
const AnalysisPage = lazy(() => import("../pages/AnalysisPage"));
const Comparison = lazy(() => import("../pages/Comparison"));
const Evidence = lazy(() => import("../pages/Evidence"));
const Tools = lazy(() => import("../pages/Tools"));
const Architecture = lazy(() => import("../pages/Architecture"));
const Settings = lazy(() => import("../pages/Settings"));
const DataPage = lazy(() => import("../pages/DataPage").then(m => ({ default: m.Observability })));
const Evaluation = lazy(() => import("../pages/DataPage").then(m => ({ default: m.Evaluation })));
const router = createBrowserRouter([{ element: <AppShell />, children: [
  { path: "/", element: <Overview /> }, { path: "/analyses/new", element: <NewAnalysis /> }, { path: "/analyses/:id", element: <AnalysisPage /> },
  { path: "/comparison", element: <Comparison /> }, { path: "/evidence", element: <Evidence /> }, { path: "/tools", element: <Tools /> },
  { path: "/observability", element: <DataPage /> }, { path: "/evaluation", element: <Evaluation /> }, { path: "/architecture", element: <Architecture /> }, { path: "/settings", element: <Settings /> },
]}]);
export function App() { return <Suspense fallback={<div className="p-6"><Skeleton /></div>}><RouterProvider router={router} /></Suspense>; }
