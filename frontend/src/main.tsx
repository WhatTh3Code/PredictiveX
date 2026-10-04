import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { useAsync } from "./components/ui";
import { About } from "./pages/About";
import { DataInsights } from "./pages/DataInsights";
import { Explain } from "./pages/Explain";
import { Overview } from "./pages/Overview";
import { Performance } from "./pages/Performance";
import { Predict } from "./pages/Predict";
import { WhatIf } from "./pages/WhatIf";
import { SessionProvider } from "./state";

const ROUTES = [
  { path: "/", label: "Overview", group: "Use", el: <Overview /> },
  { path: "/predict", label: "Predict", group: "Use", el: <Predict /> },
  { path: "/explain", label: "Explain", group: "Use", el: <Explain /> },
  { path: "/what-if", label: "What-if simulator", group: "Use", el: <WhatIf /> },
  { path: "/performance", label: "Model performance", group: "Evidence", el: <Performance /> },
  { path: "/data", label: "Data insights", group: "Evidence", el: <DataInsights /> },
  { path: "/about", label: "Methodology", group: "Evidence", el: <About /> },
] as const;

function currentPath(): string {
  const h = window.location.hash.replace(/^#/, "");
  return h === "" ? "/" : h;
}

function App() {
  const [path, setPath] = useState(currentPath());
  useEffect(() => {
    const on = () => {
      setPath(currentPath());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const route = ROUTES.find((r) => r.path === path) ?? ROUTES[0];
  useEffect(() => {
    document.title = route.path === "/" ? "PredictiveX" : `${route.label} — PredictiveX`;
  }, [route]);
  const health = useAsync(() => api.health(), []);

  return (
    <div className="shell">
      <a className="skip" href="#main-content" onClick={(e) => { e.preventDefault(); document.getElementById("main-content")?.focus(); }}>
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#/">
          <span className="brand-name">Predictive<span>X</span></span>
          <span className="brand-tag">Know the failure before it happens.</span>
        </a>
        <nav className="nav" aria-label="Main">
          {(["Use", "Evidence"] as const).map((g) => (
            <div key={g} style={{ display: "contents" }}>
              <div className="nav-group">{g === "Use" ? "Assess a machine" : "How we know"}</div>
              {ROUTES.filter((r) => r.group === g).map((r) => (
                <a key={r.path} href={`#${r.path}`} aria-current={r.path === route.path ? "page" : undefined}>
                  {r.label}
                </a>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot" data-testid="server-status">
          {health.status === "loading" && "Checking server…"}
          {health.status === "error" && (
            <><span className="status-dot" style={{ background: "var(--critical)" }} />Server unreachable</>
          )}
          {health.status === "ready" && (
            <>
              <span className="status-dot" style={{ background: health.data.model_loaded ? "var(--good)" : "var(--warning)" }} />
              {health.data.model_loaded ? "Model loaded" : "Model not loaded"}
            </>
          )}
          <div style={{ marginTop: 6 }}>ALGOTHON'26 · ALG-DATA-02</div>
        </div>
      </aside>
      <main className="main" id="main-content" tabIndex={-1}>
        {route.el}
      </main>
    </div>
  );
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <SessionProvider>
        <App />
      </SessionProvider>
    </StrictMode>,
  );
}
