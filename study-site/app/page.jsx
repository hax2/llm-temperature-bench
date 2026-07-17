"use client";

import { useMemo, useState } from "react";
import rows from "../data/results.json";

const ALL_TEMPS = [0.5, 1, 1.5, 2, 2.5, 3];
const GRAPH_TEMPS = [0.5, 1, 1.5, 2];
const PROMPTS = {
  apollo_11_facts: "Apollo 11 factual account",
  everest_facts: "Mount Everest factual account",
  constrained_lighthouse_story: "Constrained lighthouse story",
  causal_clockwork_story: "Causal clockwork story",
  contradiction_repair: "Contradiction repair",
  synthesis_city_policy: "City-policy synthesis",
};
const COLORS = [
  "#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4f46e5",
  "#15803d", "#c2410c", "#0f766e", "#9333ea", "#0369a1", "#b45309", "#4338ca", "#047857",
];

const numericMean = (values) => {
  const valid = values.filter((value) => value !== null && value !== "").map(Number).filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
};
const fmt = (value, digits = 2) => value === null || value === undefined ? "—" : Number(value).toFixed(digits);
const modelName = (id) => id
  .replace("deepseek-r1-distill-qwen-14b", "DeepSeek R1 Distill Qwen 14B")
  .replace("gemma-2-9b-base", "Gemma 2 9B Base")
  .replace("gemma-2-9b-it", "Gemma 2 9B Instruct")
  .replace("gemma-3-12b-base", "Gemma 3 12B Base")
  .replace("gemma-3-12b-it", "Gemma 3 12B Instruct")
  .replace("gemma-4-e4b-base", "Gemma 4 E4B Base")
  .replace("gemma-4-e4b-it", "Gemma 4 E4B Instruct")
  .replace("llama-3.1-8b-base", "Llama 3.1 8B Base")
  .replace("llama-3.1-8b-instruct", "Llama 3.1 8B Instruct")
  .replace("mistral-7b-v0.3-base", "Mistral 7B v0.3 Base")
  .replace("mistral-7b-v0.3-instruct", "Mistral 7B v0.3 Instruct")
  .replace("nemotron-labs-diffusion-14b", "Nemotron Diffusion 14B")
  .replace("qwen-2.5-14b-base", "Qwen 2.5 14B Base")
  .replace("qwen-2.5-14b-instruct", "Qwen 2.5 14B Instruct")
  .replace("qwen-3.5-9b-base", "Qwen 3.5 9B Base")
  .replace("qwen-3.5-9b-instruct", "Qwen 3.5 9B Instruct");

const FINDINGS = [
  {
    title: "Temperature failure is a cliff, not a gradual decline.",
    body: "Mean quality falls from 4.12 at T=0.5 to 3.13 at T=1.0, 1.30 at T=1.5, and the absolute floor of 1.0 from T=2.0 onward. For these models, temperatures above 1.5 are essentially unusable.",
    values: ["4.12", "3.13", "1.30", "1.00"],
    labels: ["T=0.5", "T=1.0", "T=1.5", "T≥2.0"],
  },
  {
    title: "T=0.5 is the safest general default.",
    body: "It was best for 15 of 16 models. The exception was Gemma 4 E4B Instruct, which improved from 6.83 at T=0.5 to 7.67 at T=1.0.",
    stat: "15 / 16",
    statLabel: "models peak at T=0.5",
  },
  {
    title: "Gemma 4 E4B Instruct is the standout.",
    body: "It was the only model still producing useful output at T=1.5, scoring 5.83, while every other model averaged 1.0. That suggests unusually strong temperature robustness—or differently calibrated logits—rather than merely slightly better general quality.",
    stat: "5.83",
    statLabel: "overall at T=1.5",
  },
  {
    title: "Instruction tuning matters enormously.",
    body: "Across seven matched model families, instruct checkpoints beat their base versions by approximately +3.3 overall points at T=0.5 and +3.2 at T=1.0. For user-facing tasks, this effect is much larger than most differences between model families.",
    values: ["+3.3", "+3.2"],
    labels: ["T=0.5", "T=1.0"],
  },
  {
    title: "More lexical diversity does not mean more creativity.",
    body: "Unique-word ratio rises from 0.37 at T=0.5 to 0.96 at T=3.0, while quality collapses to 1. Random multilingual fragments look “diverse” to lexical metrics. Likewise, repetition metrics improve as outputs become gibberish. Diversity metrics need a coherence or validity gate.",
    values: ["0.37", "0.96"],
    labels: ["unique-word ratio at T=0.5", "unique-word ratio at T=3.0"],
  },
  {
    title: "Higher temperature provided almost no useful creativity tradeoff.",
    body: "The judge’s creativity subscore rises slightly at T=1.0, but overall creative-task quality still declines from 3.09 to 2.81. The extra novelty generally does not compensate for reduced control.",
    values: ["3.09", "2.81"],
    labels: ["creative-task quality, T=0.5", "creative-task quality, T=1.0"],
  },
  {
    title: "The token cap becomes a useful warning signal.",
    body: "Cap hits rise from roughly 35% at T=0.5 to 60% at T=1.5 and 95% at T=3.0. High-temperature degeneration often fails to terminate naturally.",
    values: ["35%", "60%", "95%"],
    labels: ["T=0.5", "T=1.5", "T=3.0"],
  },
  {
    title: "Some tasks expose weakness sooner.",
    body: "The causal clockwork story was already difficult at T=0.5, whereas city-policy synthesis remained comparatively resilient at T=1.0. A benchmark containing only straightforward factual prompts would therefore overestimate robustness.",
  },
];

function Score({ value }) {
  const score = Number(value);
  const className = score >= 7 ? "high" : score >= 4 ? "medium" : "low";
  return <span className={`score ${className}`}>{fmt(score, 1)}</span>;
}

function Findings() {
  return (
    <div className="page">
      <header className="page-heading">
        <p className="kicker">Baseline findings</p>
        <h1>Temperature sensitivity in local language models</h1>
        <p>Summary of 576 judged generations across 16 models, 6 tasks, and 6 sampling temperatures.</p>
      </header>

      <blockquote className="headline">
        Across these models, temperature behaves less like a smooth creativity dial and more like a
        model-specific stability threshold. Instruction tuning raises useful quality, while Gemma 4 E4B
        Instruct uniquely shifts the collapse threshold upward.
      </blockquote>

      <section className="finding-list" aria-label="Principal findings">
        {FINDINGS.map((finding, index) => (
          <article className="finding" key={finding.title}>
            <div className="finding-number">{index + 1}</div>
            <div className="finding-text">
              <h2>{finding.title}</h2>
              <p>{finding.body}</p>
            </div>
            {(finding.stat || finding.values) && (
              <div className="finding-data">
                {finding.stat ? (
                  <div className="single-stat"><strong>{finding.stat}</strong><span>{finding.statLabel}</span></div>
                ) : finding.values.map((value, valueIndex) => (
                  <div key={finding.labels[valueIndex]}><strong>{value}</strong><span>{finding.labels[valueIndex]}</span></div>
                ))}
              </div>
            )}
          </article>
        ))}
      </section>

      <section className="method-note">
        <h2>Interpretation boundary</h2>
        <p>
          These findings describe this benchmark’s model set, loaders, prompts, decoding configuration, and judge.
          They should be treated as evidence about stability thresholds in this setup, not universal temperature
          constants for every model or serving stack.
        </p>
      </section>
    </div>
  );
}

function ModelGraph() {
  const modelIds = useMemo(() => [...new Set(rows.map((row) => row.model_id))], []);
  const [variant, setVariant] = useState("all");
  const [visible, setVisible] = useState(modelIds);
  const filteredModels = modelIds.filter((id) => {
    if (variant === "all") return true;
    const row = rows.find((item) => item.model_id === id);
    return row?.variant === variant;
  });
  const shown = filteredModels.filter((id) => visible.includes(id));
  const series = shown.map((model) => ({
    model,
    values: GRAPH_TEMPS.map((temperature) =>
      numericMean(rows.filter((row) => row.model_id === model && row.temperature === temperature).map((row) => row.overall_score)),
    ),
  }));
  const width = 1040;
  const height = 520;
  const x = (index) => 70 + index * ((width - 110) / (GRAPH_TEMPS.length - 1));
  const y = (value) => height - 55 - ((value - 1) / 9) * (height - 100);
  const toggle = (model) => setVisible((current) =>
    current.includes(model) ? current.filter((item) => item !== model) : [...current, model],
  );
  const setAll = (shouldShow) => setVisible(shouldShow ? modelIds : []);

  return (
    <div className="page">
      <header className="page-heading compact">
        <p className="kicker">Model comparison</p>
        <h1>Overall quality by temperature</h1>
        <p>The graph stops at T=2.0 because every model is at the floor from that point onward.</p>
      </header>

      <div className="graph-controls">
        <div className="button-group">
          {["all", "instruct", "base"].map((value) => (
            <button key={value} className={variant === value ? "active" : ""} onClick={() => setVariant(value)}>
              {value[0].toUpperCase() + value.slice(1)}
            </button>
          ))}
        </div>
        <div>
          <button className="link-button" onClick={() => setAll(true)}>Show all</button>
          <button className="link-button" onClick={() => setAll(false)}>Clear</button>
        </div>
      </div>

      <section className="graph-panel">
        <div className="graph-scroll">
          <svg viewBox={`0 0 ${width} ${height}`} className="model-graph" role="img" aria-label="Overall quality score by model and temperature">
            {[1, 3, 5, 7, 9].map((value) => (
              <g key={value}>
                <line x1="70" x2={width - 40} y1={y(value)} y2={y(value)} className="grid-line" />
                <text x="52" y={y(value) + 4} className="axis-text">{value}</text>
              </g>
            ))}
            {GRAPH_TEMPS.map((value, index) => (
              <text key={value} x={x(index)} y={height - 20} textAnchor="middle" className="axis-text">T={value}</text>
            ))}
            <line x1={x(3)} x2={x(3)} y1="35" y2={height - 55} className="cutoff-line" />
            {series.map(({ model, values }) => {
              const color = COLORS[modelIds.indexOf(model) % COLORS.length];
              const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
              return (
                <g key={model}>
                  <polyline points={points} fill="none" stroke={color} strokeWidth={model === "gemma-4-e4b-it" ? 4 : 2.2} opacity={model === "gemma-4-e4b-it" ? 1 : .78} />
                  {values.map((value, index) => (
                    <circle key={index} cx={x(index)} cy={y(value)} r={model === "gemma-4-e4b-it" ? 5 : 3.5} fill={color}>
                      <title>{`${modelName(model)} · T=${GRAPH_TEMPS[index]} · ${fmt(value)}`}</title>
                    </circle>
                  ))}
                </g>
              );
            })}
          </svg>
        </div>
        <p className="graph-caption">
          Each point is the mean of six tasks. Overall scores range from 1 (failure) to 10. Gemma 4 E4B
          Instruct is drawn with a heavier line.
        </p>
      </section>

      <section className="legend-panel">
        <h2>Models</h2>
        <div className="model-legend">
          {filteredModels.map((model) => {
            const active = visible.includes(model);
            return (
              <button key={model} className={active ? "active" : ""} onClick={() => toggle(model)}>
                <i style={{ background: COLORS[modelIds.indexOf(model) % COLORS.length] }} />
                <span>{modelName(model)}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="curve-table-section">
        <h2>Scores by condition</h2>
        <div className="curve-table-wrap">
          <table className="curve-table">
            <thead><tr><th>Model</th>{GRAPH_TEMPS.map((temp) => <th key={temp}>T={temp}</th>)}</tr></thead>
            <tbody>
              {series
                .sort((a, b) => b.values[0] - a.values[0])
                .map(({ model, values }) => (
                  <tr key={model}>
                    <td><i style={{ background: COLORS[modelIds.indexOf(model) % COLORS.length] }} />{modelName(model)}</td>
                    {values.map((value, index) => <td key={GRAPH_TEMPS[index]}>{fmt(value)}</td>)}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function SampleModal({ row, close }) {
  if (!row) return null;
  let problems = [];
  try { problems = JSON.parse(row.major_problems || "[]"); } catch { problems = []; }
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.currentTarget === event.target && close()}>
      <article className="modal" role="dialog" aria-modal="true">
        <button className="close" onClick={close} aria-label="Close">×</button>
        <header>
          <div><span>{modelName(row.model_id)}</span><h2>{PROMPTS[row.prompt_id]}</h2></div>
          <Score value={row.overall_score} />
        </header>
        <dl className="sample-scores">
          <div><dt>Temperature</dt><dd>{row.temperature}</dd></div>
          <div><dt>Coherence</dt><dd>{fmt(row.coherence_score, 1)}</dd></div>
          <div><dt>Factuality</dt><dd>{fmt(row.factuality_score, 1)}</dd></div>
          <div><dt>Creativity</dt><dd>{fmt(row.creativity_score, 1)}</dd></div>
          <div><dt>Consistency</dt><dd>{fmt(row.internal_consistency_score, 1)}</dd></div>
          <div><dt>Instruction following</dt><dd>{fmt(row.instruction_following_score, 1)}</dd></div>
          <div><dt>Fluency</dt><dd>{fmt(row.fluency_score, 1)}</dd></div>
          <div><dt>Words</dt><dd>{row.word_count}</dd></div>
        </dl>
        {problems.length > 0 && <div className="problem-box"><strong>Major problems noted by judge</strong><ul>{problems.map((problem) => <li key={problem}>{problem}</li>)}</ul></div>}
        <div className="output">{row.output}</div>
      </article>
    </div>
  );
}

function WritingBrowser() {
  const models = useMemo(() => [...new Set(rows.map((row) => row.model_id))], []);
  const promptIds = useMemo(() => [...new Set(rows.map((row) => row.prompt_id))], []);
  const [model, setModel] = useState("gemma-4-e4b-it");
  const [prompt, setPrompt] = useState("constrained_lighthouse_story");
  const [mode, setMode] = useState("compare");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState(null);
  const comparison = ALL_TEMPS
    .map((temperature) => rows.find((row) => row.model_id === model && row.prompt_id === prompt && row.temperature === temperature))
    .filter(Boolean);
  const library = rows
    .filter((row) => category === "all" || row.category === category)
    .filter((row) => !query || `${row.output} ${row.model_id} ${row.prompt_id}`.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => b.overall_score - a.overall_score);

  return (
    <div className="page">
      <header className="page-heading compact">
        <p className="kicker">Writing samples</p>
        <h1>Generated outputs and judge scores</h1>
        <p>Compare one model and task across temperatures, or search the complete set of 576 outputs.</p>
      </header>

      <div className="subnav">
        <button className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}>Compare temperatures</button>
        <button className={mode === "library" ? "active" : ""} onClick={() => setMode("library")}>Browse all samples</button>
      </div>

      {mode === "compare" ? (
        <>
          <div className="sample-controls">
            <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{models.map((id) => <option key={id} value={id}>{modelName(id)}</option>)}</select></label>
            <label>Task<select value={prompt} onChange={(event) => setPrompt(event.target.value)}>{promptIds.map((id) => <option key={id} value={id}>{PROMPTS[id]}</option>)}</select></label>
          </div>
          <div className="comparison-grid">
            {comparison.map((row) => (
              <article className="sample-card" key={row.temperature}>
                <header><strong>T={row.temperature}</strong><Score value={row.overall_score} /></header>
                <div className="sample-meta">{row.word_count} words · {row.hit_token_cap ? "token cap reached" : "terminated below cap"}</div>
                <p>{row.output}</p>
                <button onClick={() => setSelected(row)}>Read full output and scores</button>
              </article>
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="library-controls">
            <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search outputs, models, or task names" />
            <select value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="all">All task categories</option>
              <option value="factual">Factual</option>
              <option value="creative">Creative</option>
              <option value="reasoning">Reasoning</option>
            </select>
            <span>{library.length} results</span>
          </div>
          <div className="library-grid">
            {library.map((row) => (
              <article className="library-card" key={`${row.model_id}-${row.prompt_id}-${row.temperature}`}>
                <header><span>{PROMPTS[row.prompt_id]}</span><Score value={row.overall_score} /></header>
                <h2>{modelName(row.model_id)}</h2>
                <div className="sample-meta">T={row.temperature} · {row.word_count} words</div>
                <p>{row.output}</p>
                <button onClick={() => setSelected(row)}>Open sample</button>
              </article>
            ))}
          </div>
        </>
      )}
      <SampleModal row={selected} close={() => setSelected(null)} />
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("findings");
  return (
    <>
      <header className="site-header">
        <button className="study-title" onClick={() => setPage("findings")}>
          <strong>Temperature Study</strong><span>Baseline · July 2026</span>
        </button>
        <nav>
          <button className={page === "findings" ? "active" : ""} onClick={() => setPage("findings")}>Findings</button>
          <button className={page === "models" ? "active" : ""} onClick={() => setPage("models")}>Model graph</button>
          <button className={page === "samples" ? "active" : ""} onClick={() => setPage("samples")}>Writing browser</button>
        </nav>
        <span className="status">576 / 576 judged</span>
      </header>
      <main>
        {page === "findings" && <Findings />}
        {page === "models" && <ModelGraph />}
        {page === "samples" && <WritingBrowser />}
      </main>
      <footer>
        <span>Temperature sensitivity benchmark</span>
        <span>16 models · 6 prompts · 6 temperatures · 1 sample per condition</span>
      </footer>
    </>
  );
}
