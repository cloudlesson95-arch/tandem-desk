"use client";

import type { Claim, TraceEntry } from "@/lib/types";

type Step = {
  thought?: string | null;
  tool?: string;
  action?: string;
  args?: unknown;
  observation?: unknown;
  weak?: boolean;
  blocked?: boolean;
  rule?: string;
  proposed?: string;
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="w-20 shrink-0 text-zinc-500">{label}</span>
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}

function fmt(value: unknown): string {
  if (Array.isArray(value)) return value.map(fmt).join(" · ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "");
}

function Steps({ steps }: { steps: Step[] }) {
  return (
    <ol className="mt-1 space-y-1.5 border-l border-zinc-200 pl-3 dark:border-zinc-700">
      {steps.map((s, i) => (
        <li key={i} className="space-y-0.5">
          {s.thought && <Row label="thought">{s.thought}</Row>}
          {s.tool && (
            <Row label="tool call">
              <code className="text-teal-700 dark:text-teal-300">{s.tool}</code>({fmt(s.args)})
              {s.blocked && <span className="ml-1 text-rose-600">not in allowlist</span>}
            </Row>
          )}
          {s.observation !== undefined && (
            <Row label="observed">
              {fmt(s.observation)}
              {s.weak && <span className="ml-1 text-amber-600">weak → retry</span>}
            </Row>
          )}
          {s.rule && <Row label="rule">{<span className="text-rose-600">{s.rule}</span>}</Row>}
          {s.proposed && <Row label="proposed">{s.proposed}</Row>}
          {s.action === "answer" && <Row label="action">drafted the answer</Row>}
        </li>
      ))}
    </ol>
  );
}

function NodeDetails({ entry }: { entry: TraceEntry }) {
  const e = entry as Record<string, unknown>;
  const guard = e.guard as Record<string, unknown> | undefined;
  const redactions = e.redactions as { type: string; placeholder: string; preview: string }[] | undefined;
  const claims = e.claims as Claim[] | undefined;
  const stored = e.stored as { kind: string; content: string }[] | undefined;
  return (
    <div className="space-y-0.5">
      {guard && (
        <Row label="guard">
          {String(guard.topic)} via {String(guard.source)}
          {typeof guard.injection_score === "number" && ` · injection score ${guard.injection_score.toFixed(3)}`}
          {guard.reason ? ` · ${String(guard.reason)}` : ""}
        </Row>
      )}
      {redactions && (
        <Row label="redacted">
          {redactions.length
            ? redactions.map((r) => `${r.placeholder} ← ${r.preview}`).join(", ")
            : "nothing"}
        </Row>
      )}
      {typeof e.route === "string" && <Row label="route">{e.route}</Row>}
      {(Array.isArray(e.model) ? e.model.length > 0 : typeof e.model === "string") && (
        <Row label="model">{fmt(e.model)}</Row>
      )}
      {Array.isArray(e.recalled) && (
        <Row label="recalled">{e.recalled.length ? e.recalled.join(" · ") : "no memories"}</Row>
      )}
      {Array.isArray(e.steps) && <Steps steps={e.steps as Step[]} />}
      {typeof e.paused === "string" && <Row label="paused">{e.paused}</Row>}
      {typeof e.handoff === "string" && <Row label="handoff">general policy question → {e.handoff}</Row>}
      {typeof e.decision === "string" && (
        <Row label="decision">
          {e.decision} · token {e.token_valid ? "valid" : "invalid"}
        </Row>
      )}
      {claims && (
        <div className="space-y-0.5">
          {claims.length === 0 && <Row label="claims">none to check</Row>}
          {claims.map((c, i) => (
            <Row key={i} label={c.supported ? "✓ claim" : "✗ claim"}>
              <span className={c.supported ? "" : "text-rose-600"}>
                {c.claim}{" "}
                {(c.evidence_ids ?? (c.evidence_id ? [c.evidence_id] : [])).map((id) => (
                  <code key={id}>[{id}]</code>
                ))}
                {c.reason && ` (${c.reason})`}
              </span>
            </Row>
          ))}
          {Array.isArray(e.problems) && e.problems.length > 0 && (
            <Row label="problems">{e.problems.join("; ")}</Row>
          )}
          <Row label="result">{e.blocked ? "blocked, fallback shown" : "passed"}</Row>
        </div>
      )}
      {typeof e.skipped === "string" && <Row label="skipped">{e.skipped}</Row>}
      {stored && (
        <Row label="stored">
          {stored.length ? stored.map((m) => `${m.kind}: ${m.content}`).join(" · ") : "nothing new"}
        </Row>
      )}
    </div>
  );
}

export default function TracePanel({ trace, title = "Trace" }: { trace: TraceEntry[]; title?: string }) {
  if (!trace?.length) return null;
  const total = trace.reduce((sum, t) => sum + (t.ms || 0), 0);
  return (
    <details className="mt-2 rounded-md border border-zinc-200 bg-zinc-50 text-xs dark:border-zinc-800 dark:bg-zinc-900/60">
      <summary className="cursor-pointer select-none px-3 py-1.5 text-zinc-600 dark:text-zinc-400">
        {title} · {trace.length} nodes · {(total / 1000).toFixed(1)} s
      </summary>
      <ol className="space-y-2 px-3 pb-3">
        {trace.map((t, i) => (
          <li key={i}>
            <div className="flex items-baseline justify-between font-medium">
              <code>{t.node}</code>
              <span className="tabular-nums text-zinc-500">{t.ms} ms</span>
            </div>
            <NodeDetails entry={t} />
          </li>
        ))}
      </ol>
    </details>
  );
}
