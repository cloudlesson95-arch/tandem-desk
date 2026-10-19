export type Rental = {
  id: string;
  bike: string;
  status: string;
  start_date: string;
  end_date: string;
  daily_rate: number;
};

export type Customer = { id: string; name: string; tier: string; rentals: Rental[] };

export type TraceEntry = { node: string; ms: number; [key: string]: unknown };

export type PendingAction = {
  action_id: number;
  action: string;
  args: Record<string, unknown>;
  summary: string;
  before: Rental;
  after: Rental;
  token?: string;
};

export type Claim = {
  claim: string;
  supported: boolean;
  evidence_id: string | null;
  reason: string | null;
};

export type Verification = { claims: Claim[]; problems: string[]; blocked: boolean };

export type Evidence = {
  pid: string;
  kind: string;
  title: string;
  content: string;
  score?: number | null;
};

export type Redaction = { type: string; placeholder: string; preview: string };

export type Resolution = {
  status: string;
  answer: string;
  action_result?: Rental | null;
  trace: TraceEntry[];
};

export type Turn = {
  thread_id?: string | null;
  status: string;
  answer: string;
  topic?: string | null;
  pending_action: PendingAction | null;
  action_result?: Rental | null;
  verification?: Verification | null;
  redactions: Redaction[];
  evidence: Evidence[];
  trace: TraceEntry[];
  cached?: boolean;
  reason?: string;
  example?: ExampleMessage[];
};

export type ExampleMessage =
  | { role: "user"; content: string; thread: string }
  | ({ role: "assistant"; thread: string; resolution?: Resolution } & Turn);

export type Memory = { id: number; kind: string; content: string; created_at: string };

export type ActionLogEntry = {
  id: number;
  action: string;
  args: Record<string, unknown>;
  decision: string;
  result: Record<string, unknown> | null;
  created_at: string;
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/agent${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}
