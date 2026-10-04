import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { api, ApiError, type MachineInput, type Prediction } from "./api";
import { machineToDraft, validateMachine, type FieldError } from "./lib/format";

interface Session {
  draft: Record<string, string>;
  setDraft: (d: Record<string, string>) => void;
  activeCase: string | null;
  setActiveCase: (id: string | null) => void;
  prediction: (Prediction & { input: MachineInput }) | null;
  predicting: boolean;
  fieldErrors: FieldError[];
  requestError: ApiError | null;
  runPrediction: (draft?: Record<string, string>) => Promise<boolean>;
  loadCase: (id: string, input: Record<string, unknown>, run?: boolean) => Promise<void>;
}

const SessionContext = createContext<Session | null>(null);

export const DEFAULT_DRAFT: Record<string, string> = {
  type: "M",
  air_temp_k: "300",
  process_temp_k: "310",
  rotational_speed_rpm: "1500",
  torque_nm: "40",
  tool_wear_min: "100",
};

export function SessionProvider({ children }: { children: ReactNode }) {
  const [draft, setDraftState] = useState<Record<string, string>>(DEFAULT_DRAFT);
  const [activeCase, setActiveCase] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Session["prediction"]>(null);
  const [predicting, setPredicting] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<FieldError[]>([]);
  const [requestError, setRequestError] = useState<ApiError | null>(null);

  const setDraft = useCallback((d: Record<string, string>) => {
    setDraftState(d);
    setFieldErrors([]);
  }, []);

  const runPrediction = useCallback(async (override?: Record<string, string>) => {
    const d = override ?? draft;
    const { machine, errors } = validateMachine(d);
    setFieldErrors(errors);
    setRequestError(null);
    if (!machine) {
      setPrediction(null);
      return false;
    }
    setPredicting(true);
    try {
      setPrediction(await api.predict(machine));
      return true;
    } catch (e) {
      setRequestError(e instanceof ApiError ? e : new ApiError(0, "Unexpected error"));
      return false;
    } finally {
      setPredicting(false);
    }
  }, [draft]);

  const loadCase = useCallback(async (id: string, input: Record<string, unknown>, run = true) => {
    const d = machineToDraft(input);
    setDraftState(d);
    setActiveCase(id);
    if (run) await runPrediction(d);
  }, [runPrediction]);

  return (
    <SessionContext.Provider
      value={{ draft, setDraft, activeCase, setActiveCase, prediction, predicting, fieldErrors, requestError, runPrediction, loadCase }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): Session {
  const s = useContext(SessionContext);
  if (!s) throw new Error("useSession outside SessionProvider");
  return s;
}

export function navigate(path: string) {
  window.location.hash = path;
}
