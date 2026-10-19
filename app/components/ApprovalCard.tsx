"use client";

import type { PendingAction, Resolution } from "@/lib/types";
import TracePanel from "./TracePanel";

type Props = {
  pending: PendingAction;
  resolution?: Resolution;
  busy?: boolean;
  disabled?: boolean;
  onDecide?: (decision: "approve" | "reject") => void;
};

export default function ApprovalCard({ pending, resolution, busy, disabled, onDecide }: Props) {
  const { before, after } = pending;
  const changed = pending.action === "extend_rental" ? "end_date" : "status";
  return (
    <div className="mt-2 rounded-lg border border-teal-300 bg-teal-50/60 p-3 text-sm dark:border-teal-800 dark:bg-teal-950/30">
      <div className="text-xs font-semibold uppercase tracking-wide text-teal-800 dark:text-teal-300">
        Approval needed · {pending.action.replace("_", " ")}
      </div>
      <p className="mt-1">{pending.summary}</p>
      <div className="mt-2 flex items-center gap-2 text-xs">
        <span className="rounded bg-white px-2 py-1 dark:bg-zinc-900">
          {before.id} {changed.replace("_", " ")}: <b>{String(before[changed])}</b>
        </span>
        <span aria-hidden>→</span>
        <span className="rounded bg-white px-2 py-1 dark:bg-zinc-900">
          <b>{String(after[changed])}</b>
        </span>
      </div>
      {!resolution && (
        <div className="mt-3 flex gap-2">
          <button
            className="rounded-md bg-teal-700 px-3 py-1.5 text-white hover:bg-teal-800 disabled:opacity-50"
            disabled={busy || disabled}
            onClick={() => onDecide?.("approve")}
          >
            Approve
          </button>
          <button
            className="rounded-md border border-zinc-300 px-3 py-1.5 hover:bg-white disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
            disabled={busy || disabled}
            onClick={() => onDecide?.("reject")}
          >
            Reject
          </button>
          {disabled && <span className="self-center text-xs text-zinc-500">recorded example</span>}
        </div>
      )}
      {resolution && (
        <div
          className={`mt-3 rounded-md px-3 py-2 ${
            resolution.status === "done"
              ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-200"
              : "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
          }`}
        >
          <div className="font-medium">{resolution.status === "done" ? "Done" : "Not changed"}</div>
          <div>{resolution.answer}</div>
          {resolution.action_result && (
            <div className="mt-1 text-xs">
              {resolution.action_result.id} · {resolution.action_result.bike} · {resolution.action_result.status} ·{" "}
              {resolution.action_result.start_date} → {resolution.action_result.end_date}
            </div>
          )}
          <TracePanel trace={resolution.trace} title="Approval trace" />
        </div>
      )}
    </div>
  );
}
