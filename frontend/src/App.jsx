import { useState, useEffect, useRef } from "react";

const COLORS = {
  navy: "#1a2744",
  gold: "#c8a84b",
  accent: "#2563eb",
  steel: "#64748b",
  light: "#f1f5f9",
  success: "#16a34a",
  warn: "#d97706",
  danger: "#dc2626",
};

const microstructureClasses = [

  {
    code: 1,
    label: "M, B, RA",
    full: "Martensite + Bainite + Retained Austenite",
    color: "#2563eb"
  },

  {
    code: 3,
    label: "M, F, B, RA",
    full: "Martensite + Ferrite + Bainite + Retained Austenite",
    color: "#7c3aed"
  },

  {
    code: 4,
    label: "M, F, RA",
    full: "Martensite + Ferrite + Retained Austenite",
    color: "#059669"
  },

  {
    code: 5,
    label: "M, F, RA, C",
    full: "Martensite + Ferrite + RA + Carbide",
    color: "#d97706"
  },

  {
    code: 6,
    label: "M, RA",
    full: "Martensite + Retained Austenite",
    color: "#dc2626"
  },

  {
    code: 8,
    label: "M, RA, C",
    full: "Martensite + RA + Carbide",
    color: "#0891b2"
  }

];

const classMetrics = [
  { name: "M, B, RA", precision: 0.80, recall: 0.77, f1: 0.79, support: 48 },
  { name: "M, F, B, RA", precision: 0.83, recall: 0.89, f1: 0.86, support: 45 },
  { name: "M, F, RA", precision: 0.79, recall: 0.75, f1: 0.77, support: 36 },
  { name: "M, F, RA, C", precision: 0.87, recall: 0.91, f1: 0.89, support: 45 },
  { name: "M, RA", precision: 0.57, recall: 0.47, f1: 0.51, support: 43 },
  { name: "M, RA, C", precision: 0.77, recall: 0.89, f1: 0.83, support: 46 },
];

const confusionMatrix = [
  [37, 0, 2, 1, 4, 4],
  [0, 40, 1, 0, 4, 0],
  [1, 3, 27, 0, 4, 1],
  [0, 0, 0, 41, 1, 3],
  [8, 5, 3, 3, 20, 4],
  [0, 0, 1, 2, 2, 41],
];

const classLabels = ["M,B,RA", "M,F,B,RA", "M,F,RA", "M,F,RA,C", "M,RA", "M,RA,C"];

function useIntersection(ref) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVisible(true); }, { threshold: 0.1 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return visible;
}

function AnimatedBar({ value, max = 1, color, delay = 0 }) {
  const [w, setW] = useState(0);
  useEffect(() => { const t = setTimeout(() => setW(value / max * 100), delay); return () => clearTimeout(t); }, [value, max, delay]);
  return (
    <div style={{ background: "#e2e8f0", borderRadius: 4, height: 10, overflow: "hidden" }}>
      <div style={{ width: `${w}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.8s cubic-bezier(.4,0,.2,1)" }} />
    </div>
  );
}

function ConfusionMatrixViz() {
  const max = Math.max(...confusionMatrix.flat());
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ fontSize: 11, color: COLORS.steel, marginBottom: 8, textAlign: "center" }}>Predicted Label →</div>
      <table style={{ borderCollapse: "collapse", margin: "0 auto", fontSize: 11 }}>
        <thead>
          <tr>
            <th style={{ padding: "4px 6px", color: COLORS.steel, fontSize: 10 }}>Actual ↓</th>
            {classLabels.map(l => <th key={l} style={{ padding: "4px 6px", color: COLORS.navy, fontWeight: 600, fontSize: 10 }}>{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {confusionMatrix.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: "4px 6px", fontWeight: 600, fontSize: 10, color: COLORS.navy }}>{classLabels[i]}</td>
              {row.map((val, j) => {
                const isCorrect = i === j;
                const intensity = val / max;
                const bg = isCorrect
                  ? `rgba(37,99,235,${0.15 + intensity * 0.7})`
                  : intensity > 0.3
                  ? `rgba(220,38,38,${intensity * 0.6})`
                  : `rgba(100,116,139,${intensity * 0.3})`;
                const textColor = intensity > 0.5 && isCorrect ? "#fff" : COLORS.navy;
                return (
                  <td key={j} style={{ background: bg, padding: "5px 8px", textAlign: "center", fontWeight: isCorrect ? 700 : 400, color: textColor, border: "1px solid rgba(255,255,255,0.5)", fontSize: 12, transition: "background 0.3s" }}>{val}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 12, fontSize: 11, color: COLORS.steel }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 12, height: 12, borderRadius: 2, background: "rgba(37,99,235,0.7)", display: "inline-block" }} /> Correct prediction</span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 12, height: 12, borderRadius: 2, background: "rgba(220,38,38,0.5)", display: "inline-block" }} /> Misclassified</span>
      </div>
    </div>
  );
}

function PredictionTool() {
  const [inputs, setInputs] = useState({ C: 0.29, Si: 1.35, Mn: 2.11, TMAE: 0.02, Ac1: 742, Ac3: 829, Ms: 357, QT: 229, PT: 367 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [grokSuggestion, setGrokSuggestion] = useState(null);

  const fields = [
    { key: "C", label: "Carbon C (wt.%)", min: 0.03, max: 1.20, step: 0.01 },
    { key: "Si", label: "Silicon Si (wt.%)", min: 0.0, max: 2.60, step: 0.05 },
    { key: "Mn", label: "Manganese Mn (wt.%)", min: 0.24, max: 6.0, step: 0.05 },
    { key: "TMAE", label: "Total Micro-alloy TMAE (wt.%)", min: 0.0, max: 0.30, step: 0.005 },
    { key: "Ac1", label: "Lower Critical Temp Ac₁ (°C)", min: 693, max: 773, step: 1 },
    { key: "Ac3", label: "Upper Critical Temp Ac₃ (°C)", min: 703, max: 917, step: 1 },
    { key: "Ms", label: "Martensite Start Temp Ms (°C)", min: 120, max: 454, step: 1 },
    { key: "QT", label: "Quenching Temperature QT (°C)", min: 10, max: 550, step: 1 },
    { key: "PT", label: "Partitioning Temperature PT (°C)", min: 150, max: 650, step: 1 },
  ];

  const handlePredict = async () => {

  setLoading(true)

  setResult(null)

  setGrokSuggestion(null)

  try {

const response = await fetch(
  `${import.meta.env.VITE_API_URL}/predict`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify(inputs)
      }
    )

    const data = await response.json()

    console.log(data)

    if (data.success) {

      const cls = microstructureClasses.find(
        item => item.code === data.prediction_code
      )

      setResult({

        cls: cls,

        confidence: data.confidence || 90

      })

      setGrokSuggestion(
        `Model predicted ${data.prediction_label} with ${data.confidence}% confidence.`
      )

    } else {

      alert(data.error)

    }

  } catch (error) {

    console.log(error)

    alert("Backend connection failed")

  }

  setLoading(false)
}

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 24 }}>
        {fields.map(f => (
          <div key={f.key} style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, padding: "14px 16px" }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: COLORS.navy, display: "block", marginBottom: 8 }}>{f.label}</label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input
                type="range" min={f.min} max={f.max} step={f.step} value={inputs[f.key]}
                onChange={e => setInputs(p => ({ ...p, [f.key]: parseFloat(e.target.value) }))}
                style={{ flex: 1 }}
              />
              <input
                type="number" min={f.min} max={f.max} step={f.step} value={inputs[f.key]}
                onChange={e => setInputs(p => ({ ...p, [f.key]: parseFloat(e.target.value) || 0 }))}
                style={{ width: 70, padding: "4px 6px", border: "1px solid #cbd5e1", borderRadius: 6, fontSize: 13, fontWeight: 600, color: COLORS.navy, textAlign: "right" }}
              />
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={handlePredict}
        disabled={loading}
        style={{ display: "block", margin: "0 auto 24px", background: loading ? "#94a3b8" : COLORS.navy, color: "#fff", border: "none", borderRadius: 10, padding: "14px 48px", fontSize: 15, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer", letterSpacing: 0.5 }}
      >
        {loading ? "Running k-NN Model..." : "Predict Microstructure"}
      </button>

      {result && (
        <div style={{ animation: "fadeIn 0.5s ease" }}>
          <div style={{ background: `linear-gradient(135deg, ${result.cls.color}15, ${result.cls.color}05)`, border: `2px solid ${result.cls.color}40`, borderRadius: 14, padding: 24, marginBottom: 20, textAlign: "center" }}>
            <div style={{ fontSize: 13, color: COLORS.steel, marginBottom: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Predicted Microstructure</div>
            <div style={{ fontSize: 32, fontWeight: 800, color: result.cls.color, marginBottom: 4 }}>{result.cls.label}</div>
            <div style={{ fontSize: 14, color: COLORS.navy, marginBottom: 16 }}>{result.cls.full}</div>
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 13, color: COLORS.steel }}>Model Confidence</span>
              <div style={{ background: "#e2e8f0", borderRadius: 999, height: 10, width: 160 }}>
                <div style={{ width: `${result.confidence}%`, height: "100%", background: result.cls.color, borderRadius: 999, transition: "width 1s ease" }} />
              </div>
              <span style={{ fontSize: 15, fontWeight: 700, color: result.cls.color }}>{result.confidence}%</span>
            </div>
          </div>

          {grokSuggestion && (
            <div style={{ background: "#fffbeb", border: "1px solid #fcd34d", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#92400e", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                <span>🤖</span> AI Insight & Engineering Suggestion
              </div>
              <div style={{ fontSize: 14, color: "#78350f", lineHeight: 1.7 }}>{grokSuggestion}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function F1Chart() {
  const canvasRef = useRef(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!loaded) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !window.Chart) return;

    new window.Chart(ctx, {
      type: "bar",
      data: {
        labels: classMetrics.map(m => m.name),
        datasets: [
          { label: "Precision", data: classMetrics.map(m => m.precision), backgroundColor: "rgba(37,99,235,0.7)", borderRadius: 4 },
          { label: "Recall", data: classMetrics.map(m => m.recall), backgroundColor: "rgba(124,58,237,0.7)", borderRadius: 4 },
          { label: "F1-Score", data: classMetrics.map(m => m.f1), backgroundColor: "rgba(5,150,105,0.7)", borderRadius: 4 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" }, title: { display: true, text: "Per-Class Performance Metrics (Test Set)", color: "#1a2744", font: { size: 13, weight: 600 } } },
        scales: {
          y: { beginAtZero: true, max: 1.05, ticks: { callback: v => (v * 100).toFixed(0) + "%" }, grid: { color: "#f1f5f9" } },
          x: { ticks: { font: { size: 10 } } },
        },
      },
    });
  }, [loaded]);

  return (
    <>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js" onLoad={() => setLoaded(true)} />
      <div style={{ position: "relative", width: "100%", height: 320 }}>
        <canvas ref={canvasRef} role="img" aria-label="Bar chart of per-class precision, recall, F1-score" />
      </div>
    </>
  );
}

function KNNChart() {
  const canvasRef = useRef(null);
  const [loaded, setLoaded] = useState(false);

  const kData = [
    { k: 2, train: 96, test: 72 }, { k: 3, train: 96, test: 73 }, { k: 4, train: 94, test: 74 },
    { k: 5, train: 97.7, test: 77.7 }, { k: 6, train: 94, test: 75 }, { k: 7, train: 93, test: 74 },
    { k: 8, train: 92, test: 73 }, { k: 9, train: 91, test: 72.5 }, { k: 10, train: 90, test: 71 },
    { k: 11, train: 89, test: 70 }, { k: 12, train: 88, test: 69 }, { k: 13, train: 87, test: 68.5 },
    { k: 14, train: 86, test: 68 }, { k: 15, train: 85.5, test: 67 },
  ];

  useEffect(() => {
    if (!loaded) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !window.Chart) return;

    new window.Chart(ctx, {
      type: "line",
      data: {
        labels: kData.map(d => d.k),
        datasets: [
          { label: "Training F1", data: kData.map(d => d.train), borderColor: "#2563eb", backgroundColor: "rgba(37,99,235,0.08)", tension: 0.4, pointRadius: 5, pointHoverRadius: 7, fill: true },
          { label: "Testing F1", data: kData.map(d => d.test), borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,0.08)", tension: 0.4, pointRadius: 5, pointHoverRadius: 7, fill: true, borderDash: [6, 3] },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" }, title: { display: true, text: "F1-Score vs Number of Neighbors (k)", color: "#1a2744", font: { size: 13, weight: 600 } }, annotation: {} },
        scales: {
          y: { beginAtZero: false, min: 60, max: 100, ticks: { callback: v => v + "%" }, grid: { color: "#f1f5f9" } },
          x: { title: { display: true, text: "k (Number of Neighbors)" } },
        },
      },
    });
  }, [loaded]);

  return (
    <>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js" onLoad={() => setLoaded(true)} />
      <div style={{ position: "relative", width: "100%", height: 300 }}>
        <canvas ref={canvasRef} role="img" aria-label="Line chart of F1-score vs k neighbors" />
      </div>
    </>
  );
}

function DataDistChart() {
  const canvasRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const rawData = [80, 13, 30, 14, 146, 65];
  const smoteData = [146, 146, 146, 146, 146, 146];

  useEffect(() => {
    if (!loaded) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !window.Chart) return;

    new window.Chart(ctx, {
      type: "bar",
      data: {
        labels: classLabels,
        datasets: [
          { label: "Raw Samples", data: rawData, backgroundColor: "rgba(100,116,139,0.7)", borderRadius: 4 },
          { label: "After SMOTE", data: smoteData, backgroundColor: "rgba(37,99,235,0.7)", borderRadius: 4 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" }, title: { display: true, text: "Class Distribution: Raw vs SMOTE Oversampled", color: "#1a2744", font: { size: 13, weight: 600 } } },
        scales: { y: { beginAtZero: true, grid: { color: "#f1f5f9" } } },
      },
    });
  }, [loaded]);

  return (
    <>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js" onLoad={() => setLoaded(true)} />
      <div style={{ position: "relative", width: "100%", height: 280 }}>
        <canvas ref={canvasRef} role="img" aria-label="Bar chart comparing raw vs SMOTE class distribution" />
      </div>
    </>
  );
}

const SectionTitle = ({ children, sub }) => (
  <div style={{ marginBottom: 32, paddingBottom: 16, borderBottom: `3px solid ${COLORS.gold}40` }}>
    <h2 style={{ fontSize: 26, fontWeight: 800, color: COLORS.navy, margin: 0 }}>{children}</h2>
    {sub && <p style={{ color: COLORS.steel, margin: "6px 0 0", fontSize: 15 }}>{sub}</p>}
  </div>
);

const Card = ({ children, style = {} }) => (
  <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 14, padding: 24, boxShadow: "0 2px 12px rgba(0,0,0,0.04)", ...style }}>
    {children}
  </div>
);

export default function App() {
  const [activeNav, setActiveNav] = useState("home");
  const [navOpen, setNavOpen] = useState(false);

  const navItems = [
    { id: "home", label: "Home" },
    { id: "prediction", label: "Predict" },
    { id: "performance", label: "Model Performance" },
    { id: "pipeline", label: "How It Works" },
    { id: "students", label: "Team" },
    { id: "documentation", label: "Documentation" },
  ];

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveNav(id);
    setNavOpen(false);
  };

  useEffect(() => {
    const handleScroll = () => {
      const sections = navItems.map(n => document.getElementById(n.id));
      const scrollY = window.scrollY + 120;
      for (let i = sections.length - 1; i >= 0; i--) {
        if (sections[i] && sections[i].offsetTop <= scrollY) {
          setActiveNav(navItems[i].id);
          break;
        }
      }
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div style={{ fontFamily: "'Segoe UI', system-ui, sans-serif", background: "#f8fafc", color: COLORS.navy, minHeight: "100vh" }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
        @keyframes shimmer { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        * { box-sizing: border-box; }
        body { margin: 0; }
        input[type=range] { accent-color: ${COLORS.navy}; }
        input[type=number] { outline: none; }
        input[type=number]:focus { border-color: ${COLORS.navy} !important; }
        section { scroll-margin-top: 70px; }
      `}</style>

      {/* NAVBAR */}
      <nav style={{ position: "sticky", top: 0, zIndex: 100, background: COLORS.navy, boxShadow: "0 2px 20px rgba(0,0,0,0.2)" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: COLORS.gold, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 14, color: COLORS.navy }}>NIT</div>
            <span style={{ color: "#fff", fontWeight: 700, fontSize: 15 }}>Q&P Steel ML</span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {navItems.map(n => (
              <button key={n.id} onClick={() => scrollTo(n.id)} style={{ background: activeNav === n.id ? COLORS.gold : "transparent", color: activeNav === n.id ? COLORS.navy : "rgba(255,255,255,0.75)", border: "none", borderRadius: 6, padding: "6px 12px", fontSize: 13, fontWeight: activeNav === n.id ? 700 : 400, cursor: "pointer", transition: "all 0.2s" }}>
                {n.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* HERO */}
      <section id="home" style={{ background: `linear-gradient(135deg, ${COLORS.navy} 0%, #2d3f6b 60%, #1e3a5f 100%)`, color: "#fff", padding: "80px 24px 90px", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundImage: "radial-gradient(circle at 20% 80%, rgba(200,168,75,0.12) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(37,99,235,0.15) 0%, transparent 50%)", pointerEvents: "none" }} />
        <div style={{ maxWidth: 1200, margin: "0 auto", position: "relative" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
            <div style={{ background: COLORS.gold, color: COLORS.navy, padding: "4px 14px", borderRadius: 999, fontSize: 12, fontWeight: 700, letterSpacing: 1 }}>NIT JALANDHAR</div>
            <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>Dept. of Metallurgical & Materials Engineering</div>
          </div>
          <h1 style={{ fontSize: 38, fontWeight: 900, lineHeight: 1.2, margin: "0 0 16px", maxWidth: 800 }}>
            Multi-Class Classification of Q&P Steel Microstructure
            <span style={{ color: COLORS.gold }}> using k-NN</span>
          </h1>
          <p style={{ fontSize: 17, color: "rgba(255,255,255,0.75)", maxWidth: 680, lineHeight: 1.7, marginBottom: 32 }}>
            A machine learning model that classifies Quenched & Partitioned steel microstructure types from compositional and heat-treatment features — trained on 348 Q&P steel samples from 107 research articles spanning 20 years.
          </p>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            {[
              { val: "97.7%", label: "Training F1-Score" },
              { val: "77.7%", label: "Testing F1-Score" },
              { val: "348", label: "Steel Samples" },
              { val: "6", label: "Microstructure Classes" },
              { val: "k=5", label: "Optimal Neighbors" },
            ].map(stat => (
              <div key={stat.label} style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, padding: "16px 22px", textAlign: "center", backdropFilter: "blur(8px)" }}>
                <div style={{ fontSize: 26, fontWeight: 800, color: COLORS.gold }}>{stat.val}</div>
                <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)", marginTop: 2 }}>{stat.label}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 32 }}>
            <button onClick={() => scrollTo("prediction")} style={{ background: COLORS.gold, color: COLORS.navy, border: "none", borderRadius: 10, padding: "14px 32px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>
              Try the Predictor →
            </button>
          </div>
        </div>
      </section>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px" }}>

        {/* PREDICTION */}
        <section id="prediction" style={{ padding: "60px 0" }}>
          <SectionTitle sub="Input your steel composition and heat treatment parameters to predict the resulting microstructure type">
            🔬 Microstructure Predictor
          </SectionTitle>
          <Card>
            <PredictionTool />
          </Card>
        </section>

        {/* MODEL PERFORMANCE */}
        <section id="performance" style={{ padding: "60px 0" }}>
          <SectionTitle sub="Detailed evaluation metrics of the k-NN classifier on the test dataset (263 samples)">
            📊 Model Performance
          </SectionTitle>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px,1fr))", gap: 16, marginBottom: 32 }}>
            {[
              { val: "97.7%", label: "Training F1", color: COLORS.accent, icon: "🏋️" },
              { val: "77.7%", label: "Test F1-Score", color: COLORS.success, icon: "✅" },
              { val: "78%", label: "Test Accuracy", color: "#7c3aed", icon: "🎯" },
              { val: "0.51", label: "Lowest F1 (M,RA)", color: COLORS.danger, icon: "⚠️" },
              { val: "0.89", label: "Best F1 (M,F,RA,C)", color: COLORS.success, icon: "🏆" },
            ].map(s => (
              <div key={s.label} style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12, padding: "16px 18px", textAlign: "center" }}>
                <div style={{ fontSize: 22 }}>{s.icon}</div>
                <div style={{ fontSize: 26, fontWeight: 800, color: s.color, marginTop: 4 }}>{s.val}</div>
                <div style={{ fontSize: 12, color: COLORS.steel, marginTop: 2 }}>{s.label}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
            <Card>
              <h3 style={{ margin: "0 0 16px", fontSize: 15, fontWeight: 700, color: COLORS.navy }}>Per-Class Metrics</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {classMetrics.map((m, i) => (
                  <div key={m.name}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.navy }}>{m.name}</span>
                      <span style={{ fontSize: 12, color: COLORS.steel }}>F1: <b style={{ color: m.f1 < 0.6 ? COLORS.danger : COLORS.success }}>{m.f1.toFixed(2)}</b></span>
                    </div>
                    <AnimatedBar value={m.f1} max={1} color={microstructureClasses[i].color} delay={i * 100} />
                  </div>
                ))}
              </div>
            </Card>
            <Card>
              <h3 style={{ margin: "0 0 16px", fontSize: 15, fontWeight: 700, color: COLORS.navy }}>Confusion Matrix (Test Set)</h3>
              <ConfusionMatrixViz />
            </Card>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <Card>
              <F1Chart />
            </Card>
            <Card>
              <KNNChart />
            </Card>
          </div>

          <Card style={{ marginTop: 24 }}>
            <DataDistChart />
          </Card>
        </section>

        {/* HOW IT WORKS */}
        <section id="pipeline" style={{ padding: "60px 0" }}>
          <SectionTitle sub="End-to-end explanation of the machine learning pipeline">
            ⚙️ How the Project Works
          </SectionTitle>

          <div style={{ display: "flex", flexDirection: "column", gap: 20, marginBottom: 40 }}>
            {[
              {
                step: "01", title: "Data Collection & Compilation",
                icon: "📚",
                desc: "A comprehensive dataset was compiled from ~107 published research articles covering ~400 Q&P steel microstructures over the last 20 years. Each record includes the steel chemical composition, Q&P heat treatment parameters, and the labeled microstructure type. After cleaning for inconsistencies and ambiguities, a final dataset of 348 × 16 was obtained.",
                detail: "Features: C, Si, Mn, Nb, V, Ti (wt.%) + critical temps (Ac₁, Ac₃, Mₛ) + process params (QT, PT)"
              },
              {
                step: "02", title: "Feature Engineering",
                icon: "🔧",
                desc: "Microalloying elements Nb, V, and Ti were individually sparse across the dataset, making them unreliable as separate features. They were aggregated into a single 'Total Microalloying Element' (TMAE) feature = Nb + V + Ti (wt.%). This reduced the feature space to 9 dimensions while preserving microalloying effects.",
                detail: "Final feature vector: [C, Si, Mn, TMAE, Ac₁, Ac₃, Mₛ, QT, PT]"
              },
              {
                step: "03", title: "Data Pre-processing",
                icon: "⚖️",
                desc: "Two key preprocessing steps were applied: (1) Standard scaling (zero-mean normalization) was applied to handle the wide numerical range from wt.% fractions to temperatures in hundreds of °C. (2) SMOTE (Synthetic Minority Oversampling Technique) was used to address severe class imbalance — some microstructure types had fewer than 15 examples versus 146 for the majority class.",
                detail: "After SMOTE: all 6 classes balanced to 146 samples each"
              },
              {
                step: "04", title: "k-NN Model Training",
                icon: "🤖",
                desc: "The k-Nearest Neighbor algorithm was chosen for its strength in multiclass classification on moderate-sized datasets. The optimal k=5 was determined via grid search. Final hyperparameters: n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=20, p=2 (Euclidean distance). The dataset was split 70/30 for training and testing.",
                detail: "Algorithm: scikit-learn KNeighborsClassifier | Train: 70% | Test: 30%"
              },
              {
                step: "05", title: "Model Evaluation",
                icon: "📊",
                desc: "Model performance was measured using precision, recall, and weighted F1-score for each of the 6 microstructure classes. Overall test F1-score: 77.7%. The {M, RA} class was the most challenging (F1=0.51) due to the metallurgical difficulty of obtaining pure martensite + retained austenite without other phases.",
                detail: "Best class: {M,F,RA,C} F1=0.89 | Hardest class: {M,RA} F1=0.51"
              },
              {
                step: "06", title: "Re-Engineering & Experimental Validation",
                icon: "🔬",
                desc: "The trained model was used inversely for re-engineering: a synthetic compositional space was explored to identify the parameter window producing the target {M, RA} microstructure. The predicted window (C≈0.68%, Si≈2.2%, Mn≈1.72%, QT≈PT≈200°C) was validated by manufacturing the HC9 steel at Tata Steel, Jamshedpur and conducting Q&P heat treatments. Optical and SEM micrographs confirmed the model's predictions.",
                detail: "Validated steel: C=0.68, Mn=1.72, Si=2.20, Nb=0.036 wt.% — Q&P at 200°C"
              },
            ].map((s, i) => (
              <div key={s.step} style={{ display: "flex", gap: 20, animation: `fadeIn 0.5s ease ${i * 0.1}s both` }}>
                <div style={{ flexShrink: 0, width: 64, height: 64, borderRadius: 16, background: COLORS.navy, color: COLORS.gold, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 15 }}>
                  {s.step}
                </div>
                <Card style={{ flex: 1, margin: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <span style={{ fontSize: 22 }}>{s.icon}</span>
                    <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: COLORS.navy }}>{s.title}</h3>
                  </div>
                  <p style={{ margin: "0 0 10px", color: COLORS.steel, fontSize: 14, lineHeight: 1.7 }}>{s.desc}</p>
                  <div style={{ background: COLORS.light, borderRadius: 8, padding: "8px 14px", fontSize: 12, fontFamily: "monospace", color: "#334155" }}>📌 {s.detail}</div>
                </Card>
              </div>
            ))}
          </div>

          {/* Architecture diagram */}
          <Card>
            <h3 style={{ margin: "0 0 20px", fontSize: 17, fontWeight: 700, color: COLORS.navy }}>ML Pipeline Overview</h3>
            <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap", justifyContent: "center" }}>
              {[
                { label: "Raw Data\n107 Papers", color: "#e0f2fe", border: "#7dd3fc", text: "#0c4a6e" },
                { label: "Feature\nMatrix 348×9", color: "#ede9fe", border: "#a78bfa", text: "#4c1d95" },
                { label: "Scaling +\nSMOTE", color: "#fef3c7", border: "#fcd34d", text: "#78350f" },
                { label: "k-NN\nClassifier", color: "#d1fae5", border: "#6ee7b7", text: "#064e3b" },
                { label: "Prediction\n6 Classes", color: "#fee2e2", border: "#fca5a5", text: "#7f1d1d" },
                { label: "Re-engineering\n& Validation", color: "#f0fdf4", border: "#86efac", text: "#14532d" },
              ].map((s, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center" }}>
                  <div style={{ background: s.color, border: `2px solid ${s.border}`, borderRadius: 12, padding: "12px 18px", textAlign: "center", minWidth: 100, fontSize: 12, fontWeight: 600, color: s.text, whiteSpace: "pre-line", lineHeight: 1.4 }}>
                    {s.label}
                  </div>
                  {i < 5 && <div style={{ color: COLORS.steel, fontSize: 20, margin: "0 2px" }}>→</div>}
                </div>
              ))}
            </div>
          </Card>
        </section>

        {/* TEAM */}
        <section id="students" style={{ padding: "60px 0" }}>
          <SectionTitle sub="Research team behind this project">
            👩‍🔬 Project Team
          </SectionTitle>

          <div style={{ marginBottom: 32 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.steel, textTransform: "uppercase", letterSpacing: 1, marginBottom: 20 }}>Project Students</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px,1fr))", gap: 20 }}>
              {[
                { name: "Ashutosh Kumar Gupta", role: "Software, Formal Analysis, Data Curation, Methodology, Visualization, Writing", inst: "NIT Raipur" },
                { name: "Sunny Chakroborty", role: "Data Curation, Investigation, Formal Analysis, Writing", inst: "IIEST Shibpur" },
                { name: "Swarup Kumar Ghosh", role: "Investigation, Supervision, Validation", inst: "IIEST Shibpur" },
              ].map(s => (
                <Card key={s.name}>
                  <div style={{ width: 48, height: 48, borderRadius: "50%", background: COLORS.navy, color: COLORS.gold, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 16, marginBottom: 12 }}>
                    {s.name.split(" ").map(n => n[0]).join("").slice(0, 2)}
                  </div>
                  <div style={{ fontWeight: 700, fontSize: 15, color: COLORS.navy }}>{s.name}</div>
                  <div style={{ fontSize: 12, color: COLORS.steel, margin: "4px 0 8px" }}>{s.inst}</div>
                  <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{s.role}</div>
                </Card>
              ))}
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.steel, textTransform: "uppercase", letterSpacing: 1, marginBottom: 20 }}>Project Mentor / Principal Investigator</h3>
            <Card style={{ maxWidth: 480 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{ width: 64, height: 64, borderRadius: "50%", background: COLORS.gold, color: COLORS.navy, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 22 }}>SG</div>
                <div>
                  <div style={{ fontWeight: 800, fontSize: 17, color: COLORS.navy }}>Prof. Subhas Ganguly</div>
                  <div style={{ fontSize: 13, color: COLORS.steel, marginTop: 2 }}>Dept. of Metallurgical & Materials Engineering</div>
                  <div style={{ fontSize: 13, color: COLORS.steel }}>NIT Raipur — 492010, India</div>
                  <div style={{ fontSize: 12, color: COLORS.accent, marginTop: 4 }}>sganguly.mme@nitrr.ac.in</div>
                </div>
              </div>
              <div style={{ borderTop: "1px solid #e2e8f0", marginTop: 16, paddingTop: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.steel, marginBottom: 6 }}>Contributions</div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Conceptualization · Methodology · Supervision · Resources · Writing – review & editing · Project Administration · Funding Acquisition</div>
                <div style={{ marginTop: 10, background: "#fffbeb", border: "1px solid #fcd34d", borderRadius: 8, padding: "8px 12px", fontSize: 12, color: "#78350f" }}>
                  💰 Funded by DST-SERB Core Research Grant — CRG/2021/005256
                </div>
              </div>
            </Card>
          </div>
        </section>

        {/* DOCUMENTATION */}
        <section id="documentation" style={{ padding: "60px 0" }}>
          <SectionTitle sub="Research references, dataset details, and technical documentation">
            📄 Documentation
          </SectionTitle>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
            <Card>
              <h3 style={{ margin: "0 0 16px", fontSize: 16, fontWeight: 700, color: COLORS.navy }}>📰 Primary Research Paper</h3>
              <div style={{ background: COLORS.light, borderRadius: 10, padding: 16, marginBottom: 14 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: COLORS.navy, marginBottom: 6 }}>A machine learning model for multi-class classification of quenched and partitioned steel microstructure type by the k-nearest neighbor algorithm</div>
                <div style={{ fontSize: 12, color: COLORS.steel }}>Ashutosh Kumar Gupta, Sunny Chakroborty, Swarup Kumar Ghosh, Subhas Ganguly</div>
                <div style={{ fontSize: 12, color: COLORS.accent, marginTop: 4 }}>Computational Materials Science 228 (2023) 112321 · Elsevier</div>
                <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>https://doi.org/10.1016/j.commatsci.2023.112321</div>
              </div>
              <div style={{ fontSize: 13, color: COLORS.steel, lineHeight: 1.7 }}>Data and Python code available on Mendeley Data, V1 at https://doi.org/10.17632/gy7kcvf98b.1</div>
            </Card>

            <Card>
              <h3 style={{ margin: "0 0 16px", fontSize: 16, fontWeight: 700, color: COLORS.navy }}>🗂️ Dataset Summary</h3>
              {[
                { label: "Total samples", value: "348" },
                { label: "Features (input)", value: "9 (C, Si, Mn, TMAE, Ac₁, Ac₃, Mₛ, QT, PT)" },
                { label: "Target classes", value: "6 microstructure types" },
                { label: "Train / Test split", value: "70% / 30%" },
                { label: "Source articles", value: "~107 peer-reviewed papers" },
                { label: "Coverage period", value: "Last 20 years of Q&P research" },
                { label: "Balancing technique", value: "SMOTE oversampling (→ 146 per class)" },
                { label: "Scaling method", value: "Zero-mean standardization (z-score)" },
              ].map(r => (
                <div key={r.label} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: "1px solid #f1f5f9", fontSize: 13 }}>
                  <span style={{ color: COLORS.steel }}>{r.label}</span>
                  <span style={{ fontWeight: 600, color: COLORS.navy, maxWidth: "55%", textAlign: "right" }}>{r.value}</span>
                </div>
              ))}
            </Card>
          </div>

          <Card style={{ marginBottom: 24 }}>
            <h3 style={{ margin: "0 0 16px", fontSize: 16, fontWeight: 700, color: COLORS.navy }}>🔑 Key Supporting References</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px,1fr))", gap: 14 }}>
              {[
                { cite: "[1]", title: "Modelling steel microstructure knowledge for in-silico recognition using ML", authors: "Gupta et al.", journal: "Mater. Chem. Phys. 252 (2020)" },
                { cite: "[9]", title: "Image driven ML methods for microstructure recognition", authors: "Chowdhury et al.", journal: "Comput. Mater. Sci. 123 (2016)" },
                { cite: "[10]", title: "Advanced microstructure classification by data mining methods", authors: "Gola et al.", journal: "Comput. Mater. Sci. 148 (2018)" },
                { cite: "[12]", title: "Advanced steel microstructural classification by deep learning", authors: "Azimi et al.", journal: "Sci. Rep. 8 (2018)" },
                { cite: "[18]", title: "Microstructure & mechanical properties of Q&P steel", authors: "Sun & Yu", journal: "Mater. Sci. Eng. A 586 (2013)" },
                { cite: "[31]", title: "Nearest neighbor pattern classification (foundational)", authors: "Cover & Hart", journal: "IEEE Trans. Inf. Theory 13 (1967)" },
              ].map(r => (
                <div key={r.cite} style={{ background: COLORS.light, borderRadius: 10, padding: "12px 14px" }}>
                  <span style={{ background: COLORS.navy, color: COLORS.gold, borderRadius: 4, padding: "2px 7px", fontSize: 11, fontWeight: 700, marginRight: 8 }}>{r.cite}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.navy }}>{r.title}</span>
                  <div style={{ fontSize: 11, color: COLORS.steel, marginTop: 4 }}>{r.authors} · {r.journal}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <h3 style={{ margin: "0 0 16px", fontSize: 16, fontWeight: 700, color: COLORS.navy }}>🛠️ Technical Stack</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {["Python 3", "scikit-learn", "pandas", "NumPy", "matplotlib", "imbalanced-learn (SMOTE)", "Jupyter Notebook", "k-NN Classifier", "Ball-Tree Algorithm", "SMOTE Oversampling", "Z-score Normalization", "Confusion Matrix Eval"].map(t => (
                <span key={t} style={{ background: "#e0f2fe", color: "#0c4a6e", border: "1px solid #bae6fd", borderRadius: 999, padding: "4px 12px", fontSize: 12, fontWeight: 600 }}>{t}</span>
              ))}
            </div>
          </Card>
        </section>

      </div>

      {/* FOOTER */}
      <footer style={{ background: COLORS.navy, color: "rgba(255,255,255,0.6)", padding: "40px 24px", marginTop: 60, textAlign: "center" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", marginBottom: 6 }}>Dr. B.R. Ambedkar National Institute of Technology, Jalandhar</div>
          <div style={{ fontSize: 13, marginBottom: 4 }}>Department of Metallurgical & Materials Engineering</div>
          <div style={{ fontSize: 13, marginBottom: 16 }}>Jalandhar — 144011, Punjab, India</div>
          <div style={{ display: "inline-block", background: "rgba(200,168,75,0.15)", border: "1px solid rgba(200,168,75,0.4)", borderRadius: 8, padding: "8px 20px", fontSize: 12, color: COLORS.gold }}>
            Funded by DST-SERB Core Research Grant CRG/2021/005256
          </div>
          <div style={{ marginTop: 16, fontSize: 11 }}>© 2023 Gupta et al. · Published in Computational Materials Science 228 (2023) 112321 · Elsevier</div>
        </div>
      </footer>
    </div>
  );
}
