import type { Metadata } from "next";
import raw from "@/eval/results.json";

export const metadata: Metadata = { title: "Evaluation · Tandem Desk" };

type Category = {
  category: string;
  n: number;
  answered_on?: number;
  hallucinated_on?: number;
  answered_off?: number;
  hallucinated_off?: number;
  behaviour_ok?: number;
};

type Results = {
  placeholder?: boolean;
  note?: string;
  generated_at?: string;
  set_size?: number;
  models?: Record<string, string>;
  hallucination_rate?: { verifier_on: number; verifier_off: number; answered_on: number; answered_off: number };
  refusal_accuracy?: { overall: number; off_topic: number; injection: number; n: number };
  pii?: { questions: number; leaks: number; redacted: number };
  blocked?: { count: number; rate: number };
  ragas?: { faithfulness: number; answer_relevancy: number; context_precision: number; n: number; judge?: string };
  categories?: Category[];
  failure?: { title: string; found: string; trace: string; fix: string; before?: string; after?: string };
  nli_calibration?: { supported_accepted: number; supported_total: number; errors_caught: number; errors_total: number };
  pre_fix_ids?: string[];
};

const results = raw as Results;

const pct = (v: number | undefined) => (v === undefined || v === null ? "–" : `${(v * 100).toFixed(1)}%`);

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="text-sm text-zinc-600 dark:text-zinc-400">{label}</div>
      <div className="mt-1 text-3xl font-semibold">{value}</div>
      {sub && <div className="mt-1 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex justify-between text-sm">
        <span className="text-zinc-700 dark:text-zinc-300">{label}</span>
        <span className="font-medium">{value.toFixed(3)}</span>
      </div>
      <div
        className="mt-1 h-2 rounded-full bg-teal-100 dark:bg-teal-950"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value}
        aria-label={label}
      >
        <div className="h-2 rounded-full bg-teal-600 dark:bg-teal-400" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      </div>
    </div>
  );
}

export default function EvalPage() {
  if (results.placeholder || !results.hallucination_rate) {
    return (
      <main className="mx-auto w-full max-w-4xl p-6">
        <h1 className="text-2xl font-semibold">Evaluation</h1>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">{results.note ?? "No results yet."}</p>
      </main>
    );
  }
  const h = results.hallucination_rate;
  const delta = (h.verifier_off - h.verifier_on) * 100;
  return (
    <main className="mx-auto w-full max-w-5xl space-y-8 p-6">
      <header>
        <h1 className="text-2xl font-semibold">Evaluation</h1>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          {results.set_size} generated questions (answerable, unanswerable, off-topic, PII and injection) run through
          the graph by an asyncio harness. A sentence counts as a hallucination when a Hugging Face NLI cross-encoder
          does not find it entailed by the retrieved passages.
          {results.generated_at && ` Generated ${results.generated_at.slice(0, 10)}.`}
        </p>
      </header>

      <section className="rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">Hallucination rate with the verifier on</div>
        <div className="mt-1 text-6xl font-semibold">{pct(h.verifier_on)}</div>
        <div className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          {pct(h.verifier_off)} with the verifier off · {delta >= 0 ? "−" : "+"}
          {Math.abs(delta).toFixed(1)} points · share of answered questions with any unsupported sentence ({h.answered_on}{" "}
          answered with the verifier on, {h.answered_off} with it off)
        </div>
        {results.nli_calibration && (
          <div className="mt-3 text-xs text-zinc-500">
            How far to trust it: on a hand-labelled check, the NLI scorer ({results.models?.nli}) accepts{" "}
            {results.nli_calibration.supported_accepted} of {results.nli_calibration.supported_total} supported claims
            and catches {results.nli_calibration.errors_caught} of {results.nli_calibration.errors_total} planted false
            claims, so the rate is a strict, approximate measure.
          </div>
        )}
      </section>

      {results.pre_fix_ids && results.pre_fix_ids.length > 0 && (
        <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          {results.pre_fix_ids.length} answers ({results.pre_fix_ids.join(", ")}) are from before the last agent fix: the
          model provider&apos;s daily token quota ran out before they could be rerun.
        </p>
      )}

      <section className="grid gap-4 sm:grid-cols-3">
        <Tile
          label="Refusal accuracy"
          value={pct(results.refusal_accuracy?.overall)}
          sub={`off-topic ${pct(results.refusal_accuracy?.off_topic)} · injection ${pct(results.refusal_accuracy?.injection)} · ${results.refusal_accuracy?.n ?? 0} questions`}
        />
        <Tile
          label="PII leaks into model inputs"
          value={String(results.pii?.leaks ?? "–")}
          sub={`${results.pii?.questions ?? 0} PII questions · ${results.pii?.redacted ?? 0} values redacted`}
        />
        <Tile
          label="Answers blocked by the verifier"
          value={pct(results.blocked?.rate)}
          sub={`${results.blocked?.count ?? 0} answers replaced with the fallback`}
        />
      </section>

      {results.ragas && (
        <section className="rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="font-semibold">RAGAS on {results.ragas.n} answerable questions</h2>
          <p className="mb-4 text-sm text-zinc-600 dark:text-zinc-400">
            Judge: {results.ragas.judge ?? results.models?.judge}. Scores range from 0 to 1.
          </p>
          <div className="space-y-3">
            <Meter label="Faithfulness" value={results.ragas.faithfulness} />
            <Meter label="Answer relevancy" value={results.ragas.answer_relevancy} />
            <Meter label="Context precision" value={results.ragas.context_precision} />
          </div>
        </section>
      )}

      {results.categories && (
        <section>
          <h2 className="mb-2 font-semibold">By question type</h2>
          <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50 text-left text-zinc-600 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Type</th>
                  <th className="px-3 py-2 text-right font-medium">Questions</th>
                  <th className="px-3 py-2 text-right font-medium">Expected behaviour</th>
                  <th className="px-3 py-2 text-right font-medium">Hallucinated, verifier on</th>
                  <th className="px-3 py-2 text-right font-medium">Hallucinated, verifier off</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {results.categories.map((c) => (
                  <tr key={c.category} className="border-t border-zinc-200 dark:border-zinc-800">
                    <td className="px-3 py-2">{c.category}</td>
                    <td className="px-3 py-2 text-right">{c.n}</td>
                    <td className="px-3 py-2 text-right">{c.behaviour_ok ?? "–"}</td>
                    <td className="px-3 py-2 text-right">
                      {c.hallucinated_on ?? 0} of {c.answered_on ?? 0}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {c.hallucinated_off ?? 0} of {c.answered_off ?? 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {results.failure && (
        <section className="rounded-xl border border-zinc-200 bg-white p-6 text-sm dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="font-semibold">A failure the eval found: {results.failure.title}</h2>
          <dl className="mt-3 space-y-2">
            <div>
              <dt className="font-medium">What the eval showed</dt>
              <dd className="text-zinc-700 dark:text-zinc-300">{results.failure.found}</dd>
            </div>
            <div>
              <dt className="font-medium">What the trace showed</dt>
              <dd className="text-zinc-700 dark:text-zinc-300">{results.failure.trace}</dd>
            </div>
            <div>
              <dt className="font-medium">Fix</dt>
              <dd className="text-zinc-700 dark:text-zinc-300">{results.failure.fix}</dd>
            </div>
            {results.failure.before && (
              <div>
                <dt className="font-medium">Before → after</dt>
                <dd className="text-zinc-700 dark:text-zinc-300">
                  {results.failure.before} → {results.failure.after}
                </dd>
              </div>
            )}
          </dl>
        </section>
      )}

      {results.models && (
        <p className="text-xs text-zinc-500">
          Models: {Object.entries(results.models).map(([k, v]) => `${k} ${v}`).join(" · ")}
        </p>
      )}
    </main>
  );
}
