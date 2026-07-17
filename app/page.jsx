"use client";

import { useEffect, useState } from "react";
import benchmarkRows from "../data/results.json";

const TEMPS = [0.5, 1, 1.5, 2, 2.5, 3];
const METRICS = {
  overall_score: "Overall",
  coherence_score: "Coherence",
  factuality_score: "Factuality",
  creativity_score: "Creativity",
  internal_consistency_score: "Consistency",
  instruction_following_score: "Instruction following",
  fluency_score: "Fluency",
};
const PROMPT_NAMES = {
  apollo_11_facts: "Apollo 11",
  everest_facts: "Mount Everest",
  constrained_lighthouse_story: "Lighthouse story",
  causal_clockwork_story: "Clockwork story",
  contradiction_repair: "Contradiction repair",
  synthesis_city_policy: "City policy synthesis",
};
const COLORS = ["#fc6b3f", "#18a999", "#3157d5", "#9b51e0", "#e2a400", "#db3a78"];

const fmt = (value, digits = 1) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const mean = (values) => {
  const numeric = values
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map(Number)
    .filter(Number.isFinite);
  return numeric.length ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length : 0;
};
const titleCase = (value = "") =>
  value.replaceAll("-", " ").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const shortModel = (value = "") =>
  value
    .replace("deepseek-r1-distill-qwen", "DeepSeek R1 Qwen")
    .replace("mistral-7b-v0.3", "Mistral 7B")
    .replace("llama-3.1-8b", "Llama 3.1 8B")
    .replace("qwen-2.5-14b", "Qwen 2.5 14B")
    .replace("qwen-3.5-9b", "Qwen 3.5 9B")
    .replace("gemma-2-9b", "Gemma 2 9B")
    .replace("gemma-3-12b", "Gemma 3 12B")
    .replace("gemma-4-e4b", "Gemma 4 E4B")
    .replace("nemotron-labs-diffusion-14b", "Nemotron Diffusion 14B")
    .replace("-instruct", " · Instruct")
    .replace("-base", " · Base")
    .replace("-it", " · Instruct");

function Pill({ children, tone = "" }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

function Score({ value, compact = false }) {
  const score = Number(value);
  const tone = score >= 7 ? "good" : score >= 4 ? "mid" : "bad";
  return <span className={`score ${tone} ${compact ? "compact" : ""}`}>{fmt(score)}</span>;
}

function Sparkline({ values, color = "#fc6b3f", label }) {
  const width = 260;
  const height = 84;
  const points = values
    .map((value, index) => {
      const x = 8 + (index * (width - 16)) / (values.length - 1);
      const y = height - 8 - ((Number(value) - 1) / 9) * (height - 16);
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      {[1, 5, 10].map((score) => {
        const y = height - 8 - ((score - 1) / 9) * (height - 16);
        return <line key={score} x1="8" x2={width - 8} y1={y} y2={y} className="gridline" />;
      })}
      <polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinejoin="round" />
      {values.map((value, index) => {
        const [x, y] = points.split(" ")[index].split(",");
        return <circle key={index} cx={x} cy={y} r="4" fill={color} stroke="#fffdf8" strokeWidth="2" />;
      })}
    </svg>
  );
}

function TemperatureChart({ rows, metric, selectedModels, toggleModel }) {
  const models = [...new Set(rows.map((row) => row.model_id))];
  const series = models.map((model) => ({
    model,
    values: TEMPS.map((temperature) =>
      mean(rows.filter((row) => row.model_id === model && row.temperature === temperature).map((row) => row[metric])),
    ),
  }));
  const shown = selectedModels.length ? series.filter((item) => selectedModels.includes(item.model)) : series.slice(0, 5);
  const width = 900;
  const height = 380;
  const x = (index) => 65 + (index * (width - 100)) / (TEMPS.length - 1);
  const y = (value) => height - 45 - ((value - 1) / 9) * (height - 75);
  return (
    <div className="chart-wrap">
      <svg className="main-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${METRICS[metric]} by temperature`}>
        {[1, 3, 5, 7, 9].map((score) => (
          <g key={score}>
            <line x1="65" x2={width - 35} y1={y(score)} y2={y(score)} className="gridline" />
            <text x="50" y={y(score) + 4} className="axis-label">{score}</text>
          </g>
        ))}
        {TEMPS.map((temp, index) => (
          <text key={temp} x={x(index)} y={height - 18} textAnchor="middle" className="axis-label">T={temp}</text>
        ))}
        <rect x={x(2) - 48} y="30" width={width - 35 - (x(2) - 48)} height={height - 75} className="danger-zone" />
        <text x={x(2) - 34} y="50" className="danger-label">degradation zone</text>
        {shown.map((item, seriesIndex) => {
          const color = COLORS[seriesIndex % COLORS.length];
          const points = item.values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
          return (
            <g key={item.model}>
              <polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinejoin="round" />
              {item.values.map((value, index) => (
                <circle key={index} cx={x(index)} cy={y(value)} r="4" fill={color} stroke="#fffdf8" strokeWidth="2">
                  <title>{shortModel(item.model)} · T={TEMPS[index]} · {fmt(value)}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        {series.map((item, index) => {
          const active = selectedModels.includes(item.model) || (!selectedModels.length && index < 5);
          return (
            <button key={item.model} className={active ? "legend active" : "legend"} onClick={() => toggleModel(item.model)}>
              <i style={{ background: COLORS[(shown.findIndex((x) => x.model === item.model) + COLORS.length) % COLORS.length] }} />
              {shortModel(item.model)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Heatmap({ rows, metric, onChoose }) {
  const models = [...new Set(rows.map((row) => row.model_id))];
  const values = models.map((model) => ({
    model,
    scores: TEMPS.map((temperature) =>
      mean(rows.filter((row) => row.model_id === model && row.temperature === temperature).map((row) => row[metric])),
    ),
  }));
  const color = (score) => {
    const normalized = Math.max(0, Math.min(1, (score - 1) / 9));
    const hue = 10 + normalized * 145;
    return `hsl(${hue} 66% ${94 - normalized * 47}%)`;
  };
  return (
    <div className="heatmap-scroll">
      <div className="heatmap">
        <div className="heat-corner">Model</div>
        {TEMPS.map((temp) => <div className="heat-head" key={temp}>T={temp}</div>)}
        {values.map(({ model, scores }) => (
          <div className="heat-row" key={model}>
            <button className="heat-model" onClick={() => onChoose(model)}>{shortModel(model)}</button>
            {scores.map((score, index) => (
              <button
                className="heat-cell"
                style={{ background: color(score), color: score >= 6 ? "white" : "#15201f" }}
                key={TEMPS[index]}
                title={`${shortModel(model)} at T=${TEMPS[index]}: ${fmt(score, 2)}`}
                onClick={() => onChoose(model, TEMPS[index])}
              >
                {fmt(score)}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function FindingCard({ index, eyebrow, title, children, accent }) {
  return (
    <article className="finding-card" style={{ "--accent": accent }}>
      <div className="finding-index">0{index}</div>
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h3>{title}</h3>
        <p>{children}</p>
      </div>
    </article>
  );
}

function Overview({ rows, goToSamples }) {
  const byTemp = TEMPS.map((temperature) => ({
    temperature,
    score: mean(rows.filter((row) => row.temperature === temperature).map((row) => row.overall_score)),
    viable: rows.filter((row) => row.temperature === temperature && row.overall_score >= 5).length,
  }));
  const instruct = mean(rows.filter((row) => row.variant === "instruct" && row.temperature <= 1).map((row) => row.overall_score));
  const base = mean(rows.filter((row) => row.variant === "base" && row.temperature <= 1).map((row) => row.overall_score));
  const best = [...rows]
    .reduce((map, row) => {
      const key = `${row.model_id}|${row.temperature}`;
      if (!map[key]) map[key] = [];
      map[key].push(row.overall_score);
      return map;
    }, {});
  const bestCondition = Object.entries(best)
    .map(([key, values]) => ({ key, score: mean(values) }))
    .sort((a, b) => b.score - a.score)[0];
  const [bestModel, bestTemp] = bestCondition.key.split("|");
  return (
    <>
      <section className="hero-grid">
        <div>
          <p className="eyebrow coral">The baseline run · 576 judged generations</p>
          <h1>Heat makes language <em>come apart.</em></h1>
          <p className="lede">
            A field guide to how 16 local models change across six sampling temperatures—where quality holds,
            where it fractures, and what the writing actually looks like on both sides.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={() => goToSamples()}>Read the outputs <span>→</span></button>
            <a className="text-link" href="#findings">Jump to findings</a>
          </div>
        </div>
        <aside className="hero-figure">
          <p>Mean overall score</p>
          <Sparkline values={byTemp.map((item) => item.score)} label="Mean overall score falls as temperature rises" />
          <div className="temp-ticks">{TEMPS.map((temp) => <span key={temp}>{temp}</span>)}</div>
          <div className="figure-callout">
            <strong>−76%</strong>
            <span>quality from T=0.5 to T=2.0</span>
          </div>
        </aside>
      </section>

      <section className="stat-strip" aria-label="Benchmark summary">
        <div><strong>16</strong><span>models</span></div>
        <div><strong>6</strong><span>writing tasks</span></div>
        <div><strong>576</strong><span>outputs judged</span></div>
        <div><strong>0</strong><span>missing judgments</span></div>
      </section>

      <section className="section" id="findings">
        <div className="section-heading split">
          <div>
            <p className="eyebrow">What the run says</p>
            <h2>Four findings worth carrying forward</h2>
          </div>
          <p className="heading-note">Scores are Gemini judgments on a 1–10 scale. Each model–temperature point averages six distinct tasks.</p>
        </div>
        <div className="findings">
          <FindingCard index={1} eyebrow="The quality cliff" title="T=1.5 is the break point" accent="#fc6b3f">
            Mean quality drops from {fmt(byTemp[1].score, 2)} at T=1.0 to {fmt(byTemp[2].score, 2)} at T=1.5.
            Only {byTemp[2].viable} of 96 outputs still score 5 or higher.
          </FindingCard>
          <FindingCard index={2} eyebrow="The exception" title={`${shortModel(bestModel)} peaks at T=${bestTemp}`} accent="#18a999">
            The strongest condition in the run averages {fmt(bestCondition.score, 2)} overall—evidence that a modest amount
            of sampling can help a well-aligned instruct model.
          </FindingCard>
          <FindingCard index={3} eyebrow="Alignment matters" title="Instruct tuning buys resilience" accent="#3157d5">
            At T≤1.0, instruct models average {fmt(instruct, 2)} versus {fmt(base, 2)} for base models, a {fmt(instruct - base, 2)}-point advantage.
          </FindingCard>
          <FindingCard index={4} eyebrow="Universal collapse" title="Nothing survives T=2.0" accent="#9b51e0">
            Every one of the 288 outputs generated from T=2.0 through T=3.0 received the floor overall score of 1.
          </FindingCard>
        </div>
      </section>

      <section className="section explainer">
        <div>
          <p className="eyebrow">How to read this</p>
          <h2>High lexical variety is not always healthy.</h2>
        </div>
        <p>
          As models collapse into incoherent token streams, their unique-word ratio often rises while repeated n-grams fall.
          In this run, “more diverse” can mean less meaningful. Judge scores and actual outputs are the primary signal;
          repetition metrics are diagnostic context.
        </p>
      </section>
    </>
  );
}

function ModelLab({ rows, openSample }) {
  const [metric, setMetric] = useState("overall_score");
  const [variant, setVariant] = useState("all");
  const [selectedModels, setSelectedModels] = useState([]);
  const filtered = variant === "all" ? rows : rows.filter((row) => row.variant === variant);
  const models = [...new Set(filtered.map((row) => row.model_id))];
  const ranking = models
    .map((model) => {
      const modelRows = filtered.filter((row) => row.model_id === model);
      const low = modelRows.filter((row) => row.temperature <= 1);
      const curve = TEMPS.map((temperature) => mean(modelRows.filter((row) => row.temperature === temperature).map((row) => row[metric])));
      const bestIndex = curve.indexOf(Math.max(...curve));
      const lastViable = TEMPS.filter((temperature, index) => curve[index] >= 4).at(-1);
      return { model, score: mean(low.map((row) => row[metric])), curve, bestTemp: TEMPS[bestIndex], lastViable };
    })
    .sort((a, b) => b.score - a.score);
  const toggleModel = (model) =>
    setSelectedModels((current) =>
      current.includes(model) ? current.filter((item) => item !== model) : [...current.slice(-4), model],
    );
  return (
    <>
      <section className="page-intro">
        <p className="eyebrow coral">Model lab</p>
        <h1>Compare the curves, <em>not just the winners.</em></h1>
        <p className="lede">Switch metrics, isolate base and instruct variants, and click any heatmap cell to inspect its writing.</p>
      </section>
      <section className="section flush-top">
        <div className="toolbar">
          <label>Metric
            <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              {Object.entries(METRICS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <div className="segmented" aria-label="Variant filter">
            {["all", "instruct", "base"].map((value) => (
              <button key={value} className={variant === value ? "active" : ""} onClick={() => setVariant(value)}>{titleCase(value)}</button>
            ))}
          </div>
          <button className="quiet" onClick={() => setSelectedModels([])}>Reset chart</button>
        </div>
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">Temperature curve</p><h2>{METRICS[metric]} by model</h2></div>
            <p>Click model names to add or remove up to five lines.</p>
          </div>
          <TemperatureChart rows={filtered} metric={metric} selectedModels={selectedModels} toggleModel={toggleModel} />
        </div>
      </section>
      <section className="section">
        <div className="section-heading"><p className="eyebrow">The full field</p><h2>Heatmap</h2></div>
        <Heatmap rows={filtered} metric={metric} onChoose={(model, temperature = 0.5) => openSample(model, temperature)} />
      </section>
      <section className="section">
        <div className="section-heading split">
          <div><p className="eyebrow">Practical ranking</p><h2>Best performance in the usable range</h2></div>
          <p className="heading-note">Ranked by mean {METRICS[metric].toLowerCase()} across T=0.5 and T=1.0.</p>
        </div>
        <div className="ranking">
          <div className="ranking-head"><span>#</span><span>Model</span><span>Low-temp score</span><span>Best T</span><span>Last viable T</span><span>Curve</span></div>
          {ranking.map((item, index) => (
            <button className="ranking-row" key={item.model} onClick={() => openSample(item.model, item.bestTemp)}>
              <span className="rank">{String(index + 1).padStart(2, "0")}</span>
              <span><strong>{shortModel(item.model)}</strong><small>{item.model}</small></span>
              <Score value={item.score} compact />
              <span>T={item.bestTemp}</span>
              <span>{item.lastViable ? `T=${item.lastViable}` : "None"}</span>
              <Sparkline values={item.curve} label={`${shortModel(item.model)} curve`} />
            </button>
          ))}
        </div>
      </section>
    </>
  );
}

function SampleDrawer({ row, onClose }) {
  if (!row) return null;
  let problems = [];
  try { problems = JSON.parse(row.major_problems || "[]"); } catch { problems = row.major_problems ? [row.major_problems] : []; }
  const scoreMetrics = Object.entries(METRICS).filter(([key]) => key !== "overall_score");
  return (
    <div className="drawer-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Writing sample">
        <button className="drawer-close" onClick={onClose} aria-label="Close">×</button>
        <div className="drawer-head">
          <div>
            <p className="eyebrow">{PROMPT_NAMES[row.prompt_id]} · T={row.temperature}</p>
            <h2>{shortModel(row.model_id)}</h2>
          </div>
          <Score value={row.overall_score} />
        </div>
        <div className="score-grid">
          {scoreMetrics.map(([key, label]) => (
            <div key={key}><span>{label}</span><strong>{fmt(row[key])}</strong></div>
          ))}
        </div>
        {problems.length > 0 && (
          <div className="problems">
            <p className="eyebrow">Judge’s major problems</p>
            <ul>{problems.map((problem, index) => <li key={index}>{problem}</li>)}</ul>
          </div>
        )}
        <div className="sample-meta">
          <Pill>{row.category}</Pill><Pill>{row.variant}</Pill><Pill>{row.word_count} words</Pill>
          {row.hit_token_cap && <Pill tone="warn">hit token cap</Pill>}
        </div>
        <div className="full-output">{row.output}</div>
      </aside>
    </div>
  );
}

function WritingBrowser({ rows, initial, clearInitial }) {
  const models = [...new Set(rows.map((row) => row.model_id))];
  const prompts = [...new Set(rows.map((row) => row.prompt_id))];
  const [model, setModel] = useState(initial?.model || "gemma-4-e4b-it");
  const [prompt, setPrompt] = useState(initial?.prompt || "constrained_lighthouse_story");
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [mode, setMode] = useState("compare");
  useEffect(() => {
    if (initial?.model) setModel(initial.model);
    if (initial?.temperature) {
      const match = rows.find((row) => row.model_id === initial.model && row.temperature === initial.temperature);
      if (match) { setPrompt(match.prompt_id); setSelected(match); }
    }
    if (initial) clearInitial();
  }, [initial, rows, clearInitial]);
  const comparison = TEMPS.map((temperature) => rows.find(
    (row) => row.model_id === model && row.prompt_id === prompt && row.temperature === temperature,
  )).filter(Boolean);
  const filtered = rows
    .filter((row) => category === "all" || row.category === category)
    .filter((row) => !query || `${row.output} ${row.model_id} ${row.prompt_id}`.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => b.overall_score - a.overall_score);
  return (
    <>
      <section className="page-intro samples-intro">
        <p className="eyebrow coral">Writing browser</p>
        <h1>Read what the <em>numbers mean.</em></h1>
        <p className="lede">Follow one model through rising temperatures, or search all 576 outputs for patterns, phrases, and failure modes.</p>
      </section>
      <section className="section flush-top">
        <div className="browser-tabs">
          <button className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}>Temperature comparison</button>
          <button className={mode === "library" ? "active" : ""} onClick={() => setMode("library")}>All samples</button>
        </div>
        {mode === "compare" ? (
          <>
            <div className="compare-controls">
              <label>Model
                <select value={model} onChange={(event) => setModel(event.target.value)}>
                  {models.map((item) => <option value={item} key={item}>{shortModel(item)}</option>)}
                </select>
              </label>
              <label>Writing task
                <select value={prompt} onChange={(event) => setPrompt(event.target.value)}>
                  {prompts.map((item) => <option value={item} key={item}>{PROMPT_NAMES[item] || titleCase(item)}</option>)}
                </select>
              </label>
            </div>
            <div className="comparison-summary">
              <div><p className="eyebrow">Quality path</p><h2>{shortModel(model)}</h2></div>
              <div className="score-path">
                {comparison.map((row, index) => (
                  <div key={row.temperature}>
                    <span>T={row.temperature}</span><Score value={row.overall_score} compact />
                    {index < comparison.length - 1 && <i>→</i>}
                  </div>
                ))}
              </div>
            </div>
            <div className="sample-comparison">
              {comparison.map((row) => (
                <article className="sample-card" key={row.temperature}>
                  <header>
                    <div><Pill tone={row.temperature >= 1.5 ? "warn" : ""}>T={row.temperature}</Pill><span>{row.word_count} words</span></div>
                    <Score value={row.overall_score} compact />
                  </header>
                  <p>{row.output}</p>
                  <button onClick={() => setSelected(row)}>Read full output <span>↗</span></button>
                </article>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="library-tools">
              <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search output text, models, or tasks…" />
              <div className="segmented">
                {["all", "factual", "creative", "reasoning"].map((value) => (
                  <button key={value} className={category === value ? "active" : ""} onClick={() => setCategory(value)}>{titleCase(value)}</button>
                ))}
              </div>
              <span className="result-count">{filtered.length} samples</span>
            </div>
            <div className="sample-library">
              {filtered.map((row) => (
                <article className="library-card" key={`${row.model_id}-${row.prompt_id}-${row.temperature}`}>
                  <div className="library-top"><Pill>{PROMPT_NAMES[row.prompt_id]}</Pill><Score value={row.overall_score} compact /></div>
                  <h3>{shortModel(row.model_id)}</h3>
                  <p>{row.output}</p>
                  <footer><span>T={row.temperature} · {row.word_count} words</span><button onClick={() => setSelected(row)}>Open sample →</button></footer>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
      <SampleDrawer row={selected} onClose={() => setSelected(null)} />
    </>
  );
}

export default function Home() {
  const rows = benchmarkRows;
  const [view, setView] = useState("overview");
  const [sampleInitial, setSampleInitial] = useState(null);
  useEffect(() => {
    const requested = window.location.hash.slice(1);
    if (["overview", "lab", "samples"].includes(requested)) setView(requested);
  }, []);
  const openSample = (model, temperature, prompt) => {
    setSampleInitial({ model, temperature, prompt });
    setView("samples");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const navigate = (next) => {
    setView(next);
    window.history.replaceState(null, "", `#${next}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  return (
    <div>
      <header className="site-header">
        <button className="brand" onClick={() => navigate("overview")}>
          <span className="brand-mark"><i /><i /><i /></span>
          <span>Temperature<br />Bench</span>
        </button>
        <nav>
          <button className={view === "overview" ? "active" : ""} onClick={() => navigate("overview")}>Overview</button>
          <button className={view === "lab" ? "active" : ""} onClick={() => navigate("lab")}>Model lab</button>
          <button className={view === "samples" ? "active" : ""} onClick={() => navigate("samples")}>Writing browser</button>
        </nav>
        <div className="run-badge"><i /> Baseline complete</div>
      </header>
      <main>
        {view === "overview" && <Overview rows={rows} goToSamples={openSample} />}
        {view === "lab" && <ModelLab rows={rows} openSample={openSample} />}
        {view === "samples" && <WritingBrowser rows={rows} initial={sampleInitial} clearInitial={() => setSampleInitial(null)} />}
      </main>
      <footer className="site-footer">
        <div><span className="brand-mark small"><i /><i /><i /></span><strong>Temperature Bench Explorer</strong></div>
        <p>16 models · 6 tasks · 6 temperatures · 576 judged generations</p>
        <button onClick={() => navigate("samples")}>Browse the evidence →</button>
      </footer>
    </div>
  );
}
