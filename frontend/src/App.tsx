import { useState } from 'react'
import './App.css'

interface FormData {
  age: string
  sex: string
  cigs_per_day: string
  sickle_cell_genotype: string
  malaria_exposure: string
  hemoglobin_g_per_dL: string
  heart_rate_bpm: string
  cholesterol_mg_per_dL: string
  blood_pressure_upper: string
  blood_pressure_lower: string
}

interface PredictionResult {
  prediction: number
  probability: number
  uncertainty: {
    mean: number
    std: number
    lower: number
    upper: number
  }
}

const DEFAULT_FORM: FormData = {
  age: '',
  sex: '0',
  cigs_per_day: '',
  sickle_cell_genotype: 'AA',
  malaria_exposure: '0.0',
  hemoglobin_g_per_dL: '',
  heart_rate_bpm: '',
  cholesterol_mg_per_dL: '',
  blood_pressure_upper: '',
  blood_pressure_lower: '',
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function App() {
  const [form, setForm] = useState<FormData>(DEFAULT_FORM)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [warnOpen, setWarnOpen] = useState(false)

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: Number(form.age),
          sex: Number(form.sex),
          cigs_per_day: Number(form.cigs_per_day),
          sickle_cell_genotype: form.sickle_cell_genotype,
          malaria_exposure: Number(form.malaria_exposure),
          hemoglobin_g_per_dL: Number(form.hemoglobin_g_per_dL),
          heart_rate_bpm: Number(form.heart_rate_bpm),
          cholesterol_mg_per_dL: Number(form.cholesterol_mg_per_dL),
          blood_pressure_upper: Number(form.blood_pressure_upper),
          blood_pressure_lower: Number(form.blood_pressure_lower),
        }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data: PredictionResult = await res.json()
      setResult(data)
      const { std, upper, lower } = data.uncertainty
      if (std > 0.10 || (upper - lower) >= 0.50) setWarnOpen(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const highRisk = result?.prediction === 1

  return (
    <div className="app">
      <header className="app-header">
        <h1>Cardio Risk Predictor</h1>
        <p>Bayesian classifier — enter patient data to get a prediction with uncertainty estimate</p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="age">Age</label>
            <input id="age" name="age" type="number" min="1" max="120" required
              value={form.age} onChange={handleChange} placeholder="23" />
          </div>

          <div className="field">
            <label htmlFor="sex">Sex</label>
            <select id="sex" name="sex" value={form.sex} onChange={handleChange}>
              <option value="0">Female</option>
              <option value="1">Male</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="cigs_per_day">Cigarettes / Day</label>
            <input id="cigs_per_day" name="cigs_per_day" type="number" min="0" required
              value={form.cigs_per_day} onChange={handleChange} placeholder="2" />
          </div>

          <div className="field">
            <label htmlFor="sickle_cell_genotype">Sickle Cell Genotype</label>
            <select id="sickle_cell_genotype" name="sickle_cell_genotype"
              value={form.sickle_cell_genotype} onChange={handleChange}>
              <option value="AA">AA</option>
              <option value="AS">AS</option>
              <option value="SS">SS</option>
              <option value="SC">SC</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="malaria_exposure">Malaria Exposure</label>
            <select id="malaria_exposure" name="malaria_exposure"
              value={form.malaria_exposure} onChange={handleChange}>
              <option value="0.0">Rare (0.0)</option>
              <option value="0.5">Recent (0.5)</option>
              <option value="1.0">Chronic (1.0)</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="hemoglobin_g_per_dL">Hemoglobin (g/dL)</label>
            <input id="hemoglobin_g_per_dL" name="hemoglobin_g_per_dL" type="number"
              step="0.1" required value={form.hemoglobin_g_per_dL}
              onChange={handleChange} placeholder="14" />
          </div>

          <div className="field">
            <label htmlFor="heart_rate_bpm">Heart Rate (BPM)</label>
            <input id="heart_rate_bpm" name="heart_rate_bpm" type="number" min="1" required
              value={form.heart_rate_bpm} onChange={handleChange} placeholder="80" />
          </div>

          <div className="field">
            <label htmlFor="cholesterol_mg_per_dL">Cholesterol (mg/dL)</label>
            <input id="cholesterol_mg_per_dL" name="cholesterol_mg_per_dL" type="number"
              step="0.1" required value={form.cholesterol_mg_per_dL}
              onChange={handleChange} placeholder="220" />
          </div>

          <div className="field">
            <label htmlFor="blood_pressure_upper">Blood Pressure — Systolic</label>
            <input id="blood_pressure_upper" name="blood_pressure_upper" type="number"
              step="0.1" required value={form.blood_pressure_upper}
              onChange={handleChange} placeholder="110" />
          </div>

          <div className="field">
            <label htmlFor="blood_pressure_lower">Blood Pressure — Diastolic</label>
            <input id="blood_pressure_lower" name="blood_pressure_lower" type="number"
              step="0.1" required value={form.blood_pressure_lower}
              onChange={handleChange} placeholder="60" />
          </div>
        </div>

        {error && <p className="error">{error}</p>}

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {result && (
        <div className="card result-card">
          <div className="result-top">
            <div>
              <p className="result-label">Cardio risk probability</p>
              <p className="result-prob">{pct(result.probability)}</p>
            </div>
            <span className={`badge ${highRisk ? 'badge-high' : 'badge-low'}`}>
              {highRisk ? 'High risk' : 'Low risk'}
            </span>
          </div>

          <div className="prob-bar-track">
            <div
              className={`prob-bar-fill ${highRisk ? 'fill-high' : 'fill-low'}`}
              style={{ width: pct(result.probability) }}
            />
          </div>

          <p className="uncertainty-label">Uncertainty (95% CI)</p>
          <div className="uncertainty-grid">
            <div className="stat-box">
              <span className="stat-name">Mean</span>
              <span className="stat-val">{pct(result.uncertainty.mean)}</span>
            </div>
            <div className="stat-box">
              <span className="stat-name">STD</span>
              <span className="stat-val">{pct(result.uncertainty.std)}</span>
            </div>
            <div className="stat-box">
              <span className="stat-name">Lower 2.5%</span>
              <span className="stat-val">{pct(result.uncertainty.lower)}</span>
            </div>
            <div className="stat-box">
              <span className="stat-name">Upper 97.5%</span>
              <span className="stat-val">{pct(result.uncertainty.upper)}</span>
            </div>
          </div>
        </div>
      )}
      {warnOpen && (
        <div className="modal-backdrop" onClick={() => setWarnOpen(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-icon">⚠</div>
            <h2>High Uncertainty</h2>
            <p>
              The model's confidence interval is wide — this prediction may be unreliable.
              Consider collecting additional data or consulting a clinician.
            </p>
            {result && (
              <ul className="modal-stats">
                <li>STD: <strong>{pct(result.uncertainty.std)}</strong></li>
                <li>CI width: <strong>{pct(result.uncertainty.upper - result.uncertainty.lower)}</strong></li>
              </ul>
            )}
            <button className="modal-close" onClick={() => setWarnOpen(false)}>
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
