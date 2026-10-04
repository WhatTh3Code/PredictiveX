import { useCallback, useEffect, useState, type ReactNode } from "react";
import { ApiError, type RiskLevel } from "../api";

export type AsyncState<T> =
  | { status: "loading"; data: T | null; error: null }
  | { status: "ready"; data: T; error: null }
  | { status: "error"; data: T | null; error: ApiError };

export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading", data: null, error: null });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => ({ status: "loading", data: s.data, error: null }));
    load()
      .then((data) => alive && setState({ status: "ready", data, error: null }))
      .catch((e: unknown) => {
        if (!alive) return;
        const err = e instanceof ApiError ? e : new ApiError(0, e instanceof Error ? e.message : "Unexpected error");
        setState((s) => ({ status: "error", data: s.data, error: err }));
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite" data-testid="loading">
      <span className="note">{label}…</span>
      <div className="skeleton wide" />
      <div className="skeleton mid" />
      <div className="skeleton short" />
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="state" role="alert" data-testid="error-state">
      <h3>{error.status === 0 ? "Server not reachable" : error.status === 503 ? "Model not available" : "Something went wrong"}</h3>
      <p>{error.message}</p>
      {error.issues.length > 0 && (
        <ul>
          {error.issues.map((i) => (
            <li key={i.field + i.message}>
              <b>{i.field}</b>: {i.message}
            </li>
          ))}
        </ul>
      )}
      {onRetry && (
        <button className="btn btn-quiet btn-small" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

/** Renders loading / error / content for one async resource. */
export function Async<T>({ state, children, label }: { state: AsyncState<T> & { reload: () => void }; children: (d: T) => ReactNode; label?: string }) {
  if (state.status === "error" && !state.data) return <ErrorState error={state.error} onRetry={state.reload} />;
  if (state.data === null) return <Loading label={label} />;
  return <>{children(state.data)}</>;
}

export function EmptyState({ title, children, testId = "empty-state" }: { title: string; children?: ReactNode; testId?: string }) {
  return (
    <div className="state" data-testid={testId}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

const RISK_ICON: Record<RiskLevel, ReactNode> = {
  LOW: <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />,
  MEDIUM: (
    <>
      <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M8 4.5v4.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="8" cy="11.3" r="1" fill="currentColor" />
    </>
  ),
  HIGH: (
    <>
      <path d="M8 1.8l6.6 11.6H1.4z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
      <path d="M8 6v3.6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="8" cy="11.6" r="0.95" fill="currentColor" />
    </>
  ),
};

export function RiskBadge({ level, size = 16 }: { level: RiskLevel; size?: number }) {
  return (
    <span className={`risk-badge risk-${level}`} data-testid="risk-badge">
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
        {RISK_ICON[level]}
      </svg>
      {level} risk
    </span>
  );
}

export function Readouts({ items }: { items: { value: ReactNode; label: string; sub?: ReactNode }[] }) {
  return (
    <div className="readouts">
      {items.map((it) => (
        <div className="readout" key={it.label}>
          <div className="readout-value">{it.value}</div>
          <div className="readout-label">{it.label}</div>
          {it.sub && <div className="readout-sub">{it.sub}</div>}
        </div>
      ))}
    </div>
  );
}

export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </header>
  );
}
