"use client";

import type { ActionLogEntry, Memory } from "@/lib/types";

type Props = {
  memories: Memory[];
  actions: ActionLogEntry[];
  onForget: (id: number) => void;
};

export default function MemoryPanel({ memories, actions, onForget }: Props) {
  const prefs = memories.filter((m) => m.kind === "preference");
  const episodes = memories.filter((m) => m.kind === "episode");
  return (
    <div className="space-y-5 text-sm">
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Preferences</h2>
        {prefs.length === 0 && <p className="mt-1 text-zinc-500">Nothing remembered yet.</p>}
        <ul className="mt-1 space-y-1">
          {prefs.map((m) => (
            <MemoryRow key={m.id} memory={m} onForget={onForget} />
          ))}
        </ul>
      </section>
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Conversation summaries</h2>
        {episodes.length === 0 && <p className="mt-1 text-zinc-500">No past conversations.</p>}
        <ul className="mt-1 space-y-1">
          {episodes.map((m) => (
            <MemoryRow key={m.id} memory={m} onForget={onForget} />
          ))}
        </ul>
      </section>
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Action log</h2>
        {actions.length === 0 && <p className="mt-1 text-zinc-500">No actions yet.</p>}
        <ul className="mt-1 space-y-1">
          {actions.map((a) => (
            <li key={a.id} className="rounded-md bg-zinc-100 px-2 py-1.5 text-xs dark:bg-zinc-800/70">
              <span className="font-medium">{a.action.replace("_", " ")}</span>{" "}
              {String(a.args.rental_id)}
              {a.args.days ? ` +${a.args.days}d` : ""}
              <span
                className={`ml-2 rounded px-1.5 py-0.5 ${
                  a.decision === "executed"
                    ? "bg-emerald-200 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100"
                    : a.decision === "pending"
                      ? "bg-amber-200 text-amber-900 dark:bg-amber-900 dark:text-amber-100"
                      : "bg-zinc-200 text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200"
                }`}
              >
                {a.decision}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function MemoryRow({ memory, onForget }: { memory: Memory; onForget: (id: number) => void }) {
  return (
    <li className="flex items-start justify-between gap-2 rounded-md bg-zinc-100 px-2 py-1.5 dark:bg-zinc-800/70">
      <span>{memory.content}</span>
      <button
        className="shrink-0 text-xs text-zinc-500 hover:text-rose-600"
        onClick={() => onForget(memory.id)}
        title="Forget this memory"
      >
        Forget
      </button>
    </li>
  );
}
