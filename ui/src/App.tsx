import { useCallback, useEffect, useState } from "react";
import { listExperiments } from "./api/experiments";
import type { ExperimentSummary } from "./api/types";
import { Deck } from "./components/Deck";

type AppView = "home" | "workspace" | "leaderboard";

function Leaderboard({ onBack }: { onBack: () => void }) {
  return (
    <main className="leaderboard-page">
      <header className="leaderboard-header">
        <button type="button" className="leaderboard-back" onClick={onBack}>
          <span aria-hidden="true">←</span>
          Home
        </button>
        <div className="leaderboard-brand">
          <span>OpenCompass</span>
          <strong>A²E: Agent Auditing Engine</strong>
        </div>
      </header>
      <section className="leaderboard-content">
        <p className="leaderboard-eyebrow">A²E Benchmark Results</p>
        <h1>Agent Capability Leaderboard</h1>
        <p className="leaderboard-intro">
          A comparative view of agent harness performance across models, benchmarks, and auditing dimensions.
        </p>
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Agent Harness</th>
                <th>Model</th>
                <th>Benchmark</th>
                <th>Correctness</th>
                <th>Safety</th>
                <th>Efficiency</th>
              </tr>
            </thead>
            <tbody>
              <tr className="leaderboard-empty-row">
                <td colSpan={7}>Leaderboard data is being prepared.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

export default function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [view, setView] = useState<AppView>("home");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const exps = await listExperiments();
      setExperiments(exps);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (view === "leaderboard") {
    return <Leaderboard onBack={() => setView("home")} />;
  }

  if (view === "home") {
    return (
      <main className="welcome-screen">
        <div className="welcome-grid" aria-hidden="true" />
        <div className="welcome-content">
          <p className="welcome-eyebrow">OpenCompass</p>
          <h1>
            A²E:
            <br />
            <span className="welcome-title-line">Agent Auditing Engine</span>
          </h1>
          <p className="welcome-overview">
            A²E is an end-to-end auditing engine for agent harnesses. It connects evaluation tasks across
            frameworks, captures standardized execution traces, and measures capabilities beyond correctness,
            including efficiency, planning, tool use, and error recovery.
          </p>
          <p className="welcome-intro">
            Explore benchmark runs, inspect individual agent traces, and compare evaluation metrics in one workspace.
            Choose a benchmark, then use the Agent and Run selectors to move between experiments.
          </p>
          <nav className="welcome-resources" aria-label="Project resources">
            <a href="https://arxiv.org/abs/2608.07346" target="_blank" rel="noreferrer">
              <span>Paper</span>
              <small>arXiv</small>
            </a>
            <a href="https://github.com/datamllab/A2E" target="_blank" rel="noreferrer">
              <span>GitHub</span>
              <small>Source</small>
            </a>
            <a
              href="https://colab.research.google.com/github/stevewithjobs/AEP/blob/yuchenyue/notebooks/a2e_quickstart.ipynb"
              target="_blank"
              rel="noreferrer"
            >
              <span>Colab</span>
              <small>Notebook</small>
            </a>
            <a href="https://huggingface.co/papers/2608.07346" target="_blank" rel="noreferrer">
              <span>HuggingFace</span>
              <small>Paper page</small>
            </a>
          </nav>
          <div className="welcome-actions">
            <button
              type="button"
              className="welcome-enter"
              onClick={() => setView("workspace")}
              disabled={loading}
            >
              {loading ? "Loading experiments..." : "Enter workspace"}
              <span aria-hidden="true">→</span>
            </button>
            <button
              type="button"
              className="welcome-enter welcome-leaderboard"
              onClick={() => setView("leaderboard")}
            >
              View leaderboard
              <span aria-hidden="true">↗</span>
            </button>
          </div>
          <div className="welcome-guide" aria-label="Workspace sections">
            <span><b>01</b> Tasks</span>
            <span><b>02</b> Traces</span>
            <span><b>03</b> Evaluation</span>
          </div>
        </div>
      </main>
    );
  }

  if (loading) {
    return (
      <div className="overlay">
        <div className="overlay-card">
          <div className="spinner" />
          <p className="overlay-msg">Loading benchmarks…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="overlay">
        <div className="overlay-card">
          <p className="overlay-msg">{error}</p>
          <button type="button" className="retry" onClick={load}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return <Deck experiments={experiments} onBack={() => setView("home")} />;
}
