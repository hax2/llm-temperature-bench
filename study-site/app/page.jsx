"use client";

import { useMemo, useState } from "react";
import rows from "../data/results.json";

const ALL_TEMPS = [1.5, 2, 2.5, 3];
const GRAPH_TEMPS = ALL_TEMPS;
const PROMPTS = {
  everest_facts: "Mount Everest factual account",
  causal_clockwork_story: "Causal clockwork story",
  contradiction_repair: "Contradiction repair",
};
const PROFILES = ["unfiltered", "top-k-20", "top-k-60", "top-p-090", "min-p-005", "combined"];
const PROFILE_NAMES = {
  unfiltered: "Unfiltered",
  "top-k-20": "Top-k 20",
  "top-k-60": "Top-k 60",
  "top-p-090": "Top-p 0.90",
  "min-p-005": "Min-p 0.05",
  combined: "Combined",
};
const PROFILE_DETAILS = {
  unfiltered: "top-p 1.0 · top-k 0",
  "top-k-20": "top-p 1.0 · top-k 20",
  "top-k-60": "top-p 1.0 · top-k 60",
  "top-p-090": "top-p 0.90 · top-k 0",
  "min-p-005": "top-p 1.0 · min-p 0.05",
  combined: "top-p 0.95 · top-k 60 · min-p 0.03",
};
const COLORS = ["#64748b", "#1d4ed8", "#0891b2", "#d97706", "#059669", "#7c3aed"];

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
    title: "Filtering can shift the collapse threshold, but it does not remove it.",
    body: "At T=1.5, the unfiltered sampler averages 1.00 while min-p 0.05 reaches 3.58. Across all samplers, however, no output at T≥2 scores 4 or higher. Filtering buys a narrow band of additional stability rather than making extreme temperatures generally usable.",
    values: ["1.00", "3.58", "0 / 216"],
    labels: ["unfiltered at T=1.5", "min-p 0.05 at T=1.5", "scores ≥4 at T≥2"],
  },
  {
    title: "Min-p 0.05 is the strongest rescue strategy at T=1.5.",
    body: "It produces the best aggregate score, 3.58, with 6 of 12 outputs scoring at least 4 and 4 scoring at least 5. Qwen 2.5 14B Instruct is the clearest success: it averages 5.33 and reaches 8 on the Everest task.",
    stat: "6 / 12",
    statLabel: "outputs score ≥4",
  },
  {
    title: "Simple nucleus sampling does not rescue this regime.",
    body: "Top-p 0.90 remains at the floor score of 1.00 at every tested temperature and hits the token cap in 83% of outputs. At these temperatures, removing only the low-probability tail is not selective enough to restore coherent generation.",
    values: ["1.00", "83%"],
    labels: ["mean overall", "token-cap rate"],
  },
  {
    title: "Instruction tuning and filtering reinforce one another.",
    body: "At T=1.5 with min-p 0.05, the two instruct checkpoints average 4.50 versus 2.67 for their base counterparts. With top-k 20, the split is 3.17 versus 1.17. The filter helps most when the underlying checkpoint is already instruction-aligned.",
    values: ["4.50", "2.67"],
    labels: ["instruct + min-p", "base + min-p"],
  },
  {
    title: "Termination control is not the same as quality control.",
    body: "The combined filter reduces token-cap hits from 25% at T=1.5 to only 8% at T=3.0, yet its mean quality still falls from 2.58 to 1.08. A sampler can make degenerate output stop cleanly without making it coherent.",
    values: ["8%", "1.08"],
    labels: ["cap hits at T=3", "overall at T=3"],
  },
  {
    title: "Top-k 20 is modest but comparatively stable.",
    body: "It never matches min-p’s T=1.5 peak, but it maintains means between 1.33 and 1.50 from T=2 through T=3 while keeping token-cap rates between 17% and 25%. It is a conservative containment strategy, not a quality rescue.",
    values: ["1.50", "1.33"],
    labels: ["overall at T=2", "overall at T=3"],
  },
  {
    title: "The apparent rescue is task-dependent.",
    body: "At T=1.5 with min-p 0.05, Everest factual writing averages 5.00 across models, contradiction repair averages 3.50, and the causal clockwork story remains substantially harder. Filter effectiveness should therefore be tested across task types, not inferred from factual prompts alone.",
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
        <p className="kicker">Sampler-pilot findings</p>
        <h1>Can sampling filters delay temperature collapse?</h1>
        <p>Summary of 288 judged generations across 4 models, 3 tasks, 4 temperatures, and 6 sampling profiles.</p>
      </header>

      <blockquote className="headline">
        Filtering can shift the stability threshold, but not abolish it. At T=1.5, min-p 0.05
        recovers useful output—especially for instruct models—while no tested sampler prevents
        broad collapse at T≥2.
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
          Each model–sampler–temperature point averages three tasks with one sample per task. The results identify
          strong patterns in this run, but do not estimate sampling variance or universal thresholds.
        </p>
      </section>
    </div>
  );
}

function SamplerGraph() {
  const modelIds = useMemo(() => [...new Set(rows.map((row) => row.model_id))], []);
  const [selectedModel, setSelectedModel] = useState("qwen-2.5-14b-instruct");
  const [visible, setVisible] = useState(PROFILES);
  const [hoveredProfile, setHoveredProfile] = useState(null);
  const shown = PROFILES.filter((profile) => visible.includes(profile));
  const series = shown.map((profile) => ({
    profile,
    values: GRAPH_TEMPS.map((temperature) =>
      numericMean(rows
        .filter((row) =>
          row.model_id === selectedModel
          && row.sampling_profile === profile
          && row.temperature === temperature,
        )
        .map((row) => row.overall_score)),
    ),
  }));
  const width = 1040;
  const height = 520;
  const x = (index) => 70 + index * ((width - 110) / (GRAPH_TEMPS.length - 1));
  const y = (value) => height - 55 - ((value - 1) / 9) * (height - 100);
  const toggle = (profile) => setVisible((current) =>
    current.includes(profile) ? current.filter((item) => item !== profile) : [...current, profile],
  );
  const setAll = (shouldShow) => setVisible(shouldShow ? PROFILES : []);

  return (
    <div className="page">
      <header className="page-heading compact">
        <p className="kicker">Sampler comparison</p>
        <h1>Which filters delay collapse?</h1>
        <p>Select a model to compare its six sampling profiles from T=1.5 through T=3.0.</p>
      </header>

      <div className="graph-controls">
        <label className="graph-selector">
          Model
          <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
            {modelIds.map((model) => <option key={model} value={model}>{modelName(model)}</option>)}
          </select>
        </label>
        <div>
          <button className="link-button" onClick={() => setAll(true)}>Show all</button>
          <button className="link-button" onClick={() => setAll(false)}>Clear</button>
        </div>
      </div>

      <section className="graph-panel">
        <div className={hoveredProfile ? "graph-hover-label visible" : "graph-hover-label"}>
          <span>{hoveredProfile ? "Sampler" : "Hover a line"}</span>
          {hoveredProfile && <strong>{PROFILE_NAMES[hoveredProfile]}</strong>}
        </div>
        <div className="graph-scroll">
          <svg viewBox={`0 0 ${width} ${height}`} className="model-graph" role="img" aria-label={`Overall quality by sampler for ${modelName(selectedModel)}`}>
            {[1, 3, 5, 7, 9].map((value) => (
              <g key={value}>
                <line x1="70" x2={width - 40} y1={y(value)} y2={y(value)} className="grid-line" />
                <text x="52" y={y(value) + 4} className="axis-text">{value}</text>
              </g>
            ))}
            {GRAPH_TEMPS.map((value, index) => (
              <text key={value} x={x(index)} y={height - 20} textAnchor="middle" className="axis-text">T={value}</text>
            ))}
            <line x1={x(1)} x2={x(1)} y1="35" y2={height - 55} className="cutoff-line" />
            {series.map(({ profile, values }) => {
              const color = COLORS[PROFILES.indexOf(profile)];
              const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
              const isHovered = hoveredProfile === profile;
              const isStandout = profile === "min-p-005";
              return (
                <g key={profile}>
                  <polyline
                    points={points}
                    fill="none"
                    stroke={color}
                    strokeWidth={isHovered ? 5 : isStandout ? 4 : 2.2}
                    opacity={hoveredProfile ? isHovered ? 1 : .14 : isStandout ? 1 : .78}
                    className="visible-curve"
                  />
                  {values.map((value, index) => (
                    <circle key={index} cx={x(index)} cy={y(value)} r={isStandout ? 5 : 3.5} fill={color} opacity={hoveredProfile ? isHovered ? 1 : .14 : 1}>
                      <title>{`${PROFILE_NAMES[profile]} · T=${GRAPH_TEMPS[index]} · ${fmt(value)}`}</title>
                    </circle>
                  ))}
                  <polyline
                    points={points}
                    fill="none"
                    stroke="transparent"
                    strokeWidth="16"
                    className="curve-hit-target"
                    role="button"
                    tabIndex="0"
                    aria-label={PROFILE_NAMES[profile]}
                    onMouseEnter={() => setHoveredProfile(profile)}
                    onMouseLeave={() => setHoveredProfile(null)}
                    onFocus={() => setHoveredProfile(profile)}
                    onBlur={() => setHoveredProfile(null)}
                  >
                    <title>{PROFILE_NAMES[profile]}</title>
                  </polyline>
                </g>
              );
            })}
          </svg>
        </div>
        <p className="graph-caption">
          Each point is the mean of three tasks. Scores range from 1 (failure) to 10. Min-p 0.05 is
          drawn with a heavier line; the dashed marker denotes T=2.0.
        </p>
      </section>

      <section className="legend-panel">
        <h2>Sampling profiles</h2>
        <div className="model-legend">
          {PROFILES.map((profile) => {
            const active = visible.includes(profile);
            return (
              <button key={profile} className={active ? "active" : ""} onClick={() => toggle(profile)}>
                <i style={{ background: COLORS[PROFILES.indexOf(profile)] }} />
                <span><strong>{PROFILE_NAMES[profile]}</strong><small>{PROFILE_DETAILS[profile]}</small></span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="curve-table-section">
        <h2>Mean scores for {modelName(selectedModel)}</h2>
        <div className="curve-table-wrap">
          <table className="curve-table">
            <thead><tr><th>Sampler</th>{GRAPH_TEMPS.map((temp) => <th key={temp}>T={temp}</th>)}</tr></thead>
            <tbody>
              {[...series]
                .sort((a, b) => b.values[0] - a.values[0])
                .map(({ profile, values }) => (
                  <tr key={profile}>
                    <td><i style={{ background: COLORS[PROFILES.indexOf(profile)] }} />{PROFILE_NAMES[profile]}</td>
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
          <div><dt>Sampler</dt><dd>{PROFILE_NAMES[row.sampling_profile]}</dd></div>
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
  const [model, setModel] = useState("qwen-2.5-14b-instruct");
  const [profile, setProfile] = useState("min-p-005");
  const [prompt, setPrompt] = useState("everest_facts");
  const [mode, setMode] = useState("compare");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState(null);
  const comparison = ALL_TEMPS
    .map((temperature) => rows.find((row) =>
      row.model_id === model
      && row.sampling_profile === profile
      && row.prompt_id === prompt
      && row.temperature === temperature,
    ))
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
        <p>Compare one model, sampler, and task across temperatures, or search all 288 filtered-run outputs.</p>
      </header>

      <div className="subnav">
        <button className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}>Compare temperatures</button>
        <button className={mode === "library" ? "active" : ""} onClick={() => setMode("library")}>Browse all samples</button>
      </div>

      {mode === "compare" ? (
        <>
          <div className="sample-controls">
            <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{models.map((id) => <option key={id} value={id}>{modelName(id)}</option>)}</select></label>
            <label>Sampler<select value={profile} onChange={(event) => setProfile(event.target.value)}>{PROFILES.map((id) => <option key={id} value={id}>{PROFILE_NAMES[id]} — {PROFILE_DETAILS[id]}</option>)}</select></label>
            <label>Task<select value={prompt} onChange={(event) => setPrompt(event.target.value)}>{promptIds.map((id) => <option key={id} value={id}>{PROMPTS[id]}</option>)}</select></label>
          </div>
          <div className="comparison-grid">
            {comparison.map((row) => (
              <article className="sample-card" key={row.temperature}>
                <header><strong>T={row.temperature}</strong><Score value={row.overall_score} /></header>
                <div className="sample-meta">{PROFILE_NAMES[row.sampling_profile]} · {row.word_count} words · {row.hit_token_cap ? "token cap reached" : "terminated below cap"}</div>
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
              <article className="library-card" key={`${row.model_id}-${row.sampling_profile}-${row.prompt_id}-${row.temperature}`}>
                <header><span>{PROMPTS[row.prompt_id]}</span><Score value={row.overall_score} /></header>
                <h2>{modelName(row.model_id)}</h2>
                <div className="sample-meta">{PROFILE_NAMES[row.sampling_profile]} · T={row.temperature} · {row.word_count} words</div>
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
          <strong>Temperature Study</strong><span>Sampler pilot · July 2026</span>
        </button>
        <nav>
          <button className={page === "findings" ? "active" : ""} onClick={() => setPage("findings")}>Findings</button>
          <button className={page === "models" ? "active" : ""} onClick={() => setPage("models")}>Sampler graph</button>
          <button className={page === "samples" ? "active" : ""} onClick={() => setPage("samples")}>Writing browser</button>
        </nav>
        <span className="status">288 / 288 judged</span>
      </header>
      <main>
        {page === "findings" && <Findings />}
        {page === "models" && <SamplerGraph />}
        {page === "samples" && <WritingBrowser />}
      </main>
      <footer>
        <span>High-temperature sampler comparison</span>
        <span>4 models · 3 prompts · 4 temperatures · 6 sampling profiles</span>
      </footer>
    </>
  );
}
