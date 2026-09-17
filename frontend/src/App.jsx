import React, { useState } from 'react'
import axios from 'axios'
import './App.css'

const STANCE_COLOR = {
  SUPPORTS:  'var(--support)',
  REFUTES:   'var(--refute)',
  NEUTRAL:   'var(--neutral)',
  UNRELATED: 'var(--muted)',
}

const VERDICT_LABEL = {
  SUPPORTED:    { label: 'SUPPORTED',    color: 'var(--support)' },
  REFUTED:      { label: 'REFUTED',      color: 'var(--refute)'  },
  MIXED:        { label: 'MIXED',        color: 'var(--neutral)' },
  INSUFFICIENT: { label: 'INSUFFICIENT', color: 'var(--muted)'   },
}

function ConfidenceBar({ value, color }) {
  return (
    <div className="conf-track">
      <div className="conf-fill" style={{ width: `${value * 100}%`, background: color }} />
      <span className="conf-label">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

function EvidenceCard({ item, index }) {
  const [expanded, setExpanded] = useState(false)
  const color = STANCE_COLOR[item.stance] || 'var(--muted)'
  const hostname = (() => { try { return new URL(item.source_url).hostname.replace('www.', '') } catch { return item.source_url } })()

  return (
    <div className="ev-card" style={{ '--stance-color': color }}>
      <div className="ev-header" onClick={() => setExpanded(e => !e)}>
        <span className="ev-index">#{String(index + 1).padStart(2, '0')}</span>
        <span className="ev-stance" style={{ color }}>{item.stance}</span>
        <span className="ev-domain">{hostname}</span>
        <span className="ev-cred">cred {(item.credibility_score * 100).toFixed(0)}</span>
        <span className="ev-toggle">{expanded ? '−' : '+'}</span>
      </div>
      {expanded && (
        <div className="ev-body">
          <p className="ev-excerpt">"{item.excerpt}"</p>
          {item.stance_sentence && item.stance_sentence !== item.excerpt && (
            /* The exact sentence the NLI model scored, so the badge above is
               traceable to text rather than taken on faith. */
            <p className="ev-deciding" style={{ borderColor: color }}>
              <span className="ev-deciding-label">scored sentence</span>
              {item.stance_sentence}
            </p>
          )}
          <div className="ev-meta">
            <span>confidence</span>
            <ConfidenceBar value={item.stance_confidence} color={color} />
            <span>uncertainty</span>
            <ConfidenceBar value={item.uncertainty} color="var(--dim)" />
          </div>
          <a className="ev-link" href={item.source_url} target="_blank" rel="noreferrer">
            ↗ {item.source_title}
          </a>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [claim, setClaim]     = useState('')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [elapsed, setElapsed] = useState(null)

  async function verify() {
    if (!claim.trim() || claim.length < 10) return
    setLoading(true)
    setResult(null)
    setError(null)
    const t0 = Date.now()
    try {
      const res = await axios.post('/verify', { claim, max_sources: 8 })
      setResult(res.data)
      setElapsed(((Date.now() - t0) / 1000).toFixed(1))
    } catch (e) {
      setError(e.response?.data?.detail || 'Pipeline error')
    } finally {
      setLoading(false)
    }
  }

  function onKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); verify() }
  }

  const verdict = result ? VERDICT_LABEL[result.verdict] : null

  return (
    <div className="app">
      <header className="header">
        <div className="logo">PARALLEL<span>AI</span></div>
        <p className="tagline">neural claim verification · deberta-v3 · bayesian synthesis</p>
      </header>

      <main className="main">
        <div className="input-row">
          <textarea
            className="claim-input"
            placeholder="enter a claim to verify..."
            value={claim}
            onChange={e => setClaim(e.target.value)}
            onKeyDown={onKey}
            rows={2}
          />
          <button className="verify-btn" onClick={verify} disabled={loading || claim.length < 10}>
            {loading ? <span className="spinner" /> : 'VERIFY'}
          </button>
        </div>

        {error && <div className="error">⚠ {error}</div>}

        {result && (
          <div className="result">
            <div className="verdict-row">
              <div className="verdict-badge" style={{ color: verdict.color, borderColor: verdict.color }}>
                {verdict.label}
              </div>
              <div className="verdict-stats">
                <span className="stat">
                  <span className="stat-label">confidence</span>
                  <span className="stat-val" style={{ color: verdict.color }}>{(result.confidence * 100).toFixed(0)}%</span>
                </span>
                <span className="stat">
                  <span className="stat-label">uncertainty</span>
                  <span className="stat-val">{(result.uncertainty * 100).toFixed(0)}%</span>
                </span>
                <span className="stat">
                  <span className="stat-label">sources</span>
                  <span className="stat-val">{result.sources_analyzed}</span>
                </span>
                <span className="stat">
                  <span className="stat-label">time</span>
                  <span className="stat-val">{elapsed}s</span>
                </span>
              </div>
            </div>

            <div className="ratio-bar">
              <div className="ratio-seg" style={{ width: `${result.support_ratio * 100}%`, background: 'var(--support)' }} />
              <div className="ratio-seg" style={{ width: `${result.neutral_ratio * 100}%`, background: 'var(--neutral)' }} />
              <div className="ratio-seg" style={{ width: `${result.refute_ratio  * 100}%`, background: 'var(--refute)'  }} />
            </div>
            <div className="ratio-legend">
              <span style={{ color: 'var(--support)' }}>■ support {(result.support_ratio * 100).toFixed(0)}%</span>
              <span style={{ color: 'var(--neutral)' }}>■ neutral {(result.neutral_ratio * 100).toFixed(0)}%</span>
              <span style={{ color: 'var(--refute)'  }}>■ refute  {(result.refute_ratio  * 100).toFixed(0)}%</span>
            </div>

            <p className="summary">{result.consensus_summary}</p>

            {result.mock_search && (
              /* A mock-backed result must never be mistaken for a live one. */
              <div className="mock-banner">
                offline corpus — set TAVILY_API_KEY in .env for live web search
              </div>
            )}

            <div className="evidence-list">
              <div className="ev-section-label">EVIDENCE · {result.evidence.length} sources</div>
              {result.evidence.map((item, i) => (
                <EvidenceCard key={i} item={item} index={i} />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
