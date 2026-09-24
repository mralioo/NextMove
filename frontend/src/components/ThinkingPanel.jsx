import React from 'react';
import { Brain, Fingerprint, Wrench, Zap, Clock, Sparkles, Loader2, CheckCircle2, Settings } from 'lucide-react';

const ICON_MAP = {
  brain: Brain,
  fingerprint: Fingerprint,
  tools: Wrench,
  execute: Zap,
  history: Clock,
  llm: Sparkles
};

export default function ThinkingPanel({ steps, visible }) {
  if (!visible) return null;

  const total = steps.length;
  const completed = steps.filter(s => s.status === 'done').length;
  const progress = total === 0 ? 0 : (completed / total) * 100;
  const allDone = completed === total && total > 0;

  return (
    <div className="card" style={{ width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column', padding: '16px', gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--teal)' }}>
        <Settings size={18} className={!allDone ? "spinner" : ""} style={{ animation: !allDone ? 'spin 3s linear infinite' : 'none' }} />
        <h3 style={{ fontSize: '0.9rem', margin: 0 }}>Agent Reasoning</h3>
      </div>
      
      <div style={{ height: '4px', background: 'var(--surface2)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ height: '100%', background: 'var(--teal)', width: `${progress}%`, transition: 'width 0.3s ease' }} />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {steps.map((step, idx) => {
          const Icon = ICON_MAP[step.icon] || Settings;
          const isPending = step.status === 'pending';
          const isActive = step.status === 'active';
          const isDone = step.status === 'done';

          return (
            <div key={idx} className={`thinking-step ${step.status}`} style={{ display: 'flex', gap: '12px', opacity: isPending ? 0.5 : 1, animationDelay: `${idx * 0.1}s` }}>
              <div style={{ paddingTop: '2px' }}>
                {isPending && <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: 'var(--surface2)', border: '1px solid var(--border)' }} />}
                {isActive && <Loader2 size={16} color="var(--teal)" className="spinner" />}
                {isDone && <CheckCircle2 size={16} color="var(--teal)" className="check-icon" />}
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Icon size={14} color={isActive ? 'var(--teal)' : 'var(--text-muted)'} />
                  <span className="step-text" style={{ fontSize: '0.8rem', color: isActive ? 'var(--text)' : 'var(--text-muted)' }}>
                    {step.step}
                  </span>
                </div>
                {isDone && step.detail && (
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    {step.detail}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {allDone && (
        <div style={{ marginTop: 'auto', textAlign: 'center', padding: '8px', background: 'var(--teal-dim)', borderRadius: 'var(--radius-sm)', color: 'var(--teal)', fontSize: '0.8rem' }}>
          Analysis complete ({total} steps)
        </div>
      )}
    </div>
  );
}
