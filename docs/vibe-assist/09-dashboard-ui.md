# Vibe Assist 09 — Dashboard UI
## React Doctor Dashboard with Real-Time Glucose Charts

> **Vibe Coding Session 09/10** | Estimated time: 6 hours
> Goal: A stunning, clinically professional React dashboard that wows the judges.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ React 18 + Vite app with dark clinical theme
- ✅ Patient selector with risk-tier colour badges
- ✅ Real-time glucose chart with prediction band (Recharts)
- ✅ SHAP waterfall chart for alert explanations
- ✅ Wearable metrics cards (HRV, steps, sleep)
- ✅ Alert panel with tiered severity styling
- ✅ WebSocket hook for real-time updates
- ✅ Mobile-responsive layout

---

## Design System

```
Color Palette (Dark Clinical Theme):
  Background:   #0a0f1e  (deep navy)
  Surface:      #131928  (card background)
  Border:       #1e2a3d  (subtle border)
  
  Text Primary: #e8f0fe  (soft white)
  Text Muted:   #7a8ab0  (grey-blue)
  
  Brand Accent: #4a9eff  (medical blue)
  
  Risk Colors:
    GREEN:  #00d084  (safe)
    YELLOW: #ffd600  (caution)
    ORANGE: #ff8c00  (warning)
    RED:    #ff3d00  (critical)
  
  Chart Colors:
    Glucose Historical: #4a9eff
    Glucose Predicted:  #ff8c00 (orange band)
    Normal Range Band:  rgba(0, 208, 132, 0.1)

Typography:
  Font: 'Inter', sans-serif (Google Fonts)
  Headings: font-weight 600
  Data: 'JetBrains Mono' for numbers

Layout:
  Sidebar (patients list): 280px fixed
  Main content: flex-fill
  Grid: 12-column CSS Grid
```

---

## 🤖 Vibe Prompt 1 — React App Scaffold + Design System

```
Create a React 18 + Vite dashboard for GlucoTwin with a premium 
dark clinical theme.

Create src/dashboard/src/index.css with:

:root {
  --bg-primary: #0a0f1e;
  --bg-surface: #131928;
  --bg-elevated: #1a2233;
  --border-color: #1e2a3d;
  --border-accent: #2a3a5e;
  
  --text-primary: #e8f0fe;
  --text-secondary: #b0bee0;
  --text-muted: #7a8ab0;
  
  --brand-blue: #4a9eff;
  --brand-blue-dim: rgba(74, 158, 255, 0.15);
  
  --risk-green: #00d084;
  --risk-yellow: #ffd600;
  --risk-orange: #ff8c00;
  --risk-red: #ff3d00;
  
  --risk-green-bg: rgba(0, 208, 132, 0.1);
  --risk-yellow-bg: rgba(255, 214, 0, 0.1);
  --risk-orange-bg: rgba(255, 140, 0, 0.1);
  --risk-red-bg: rgba(255, 61, 0, 0.1);
  
  --glucose-line: #4a9eff;
  --glucose-forecast: rgba(255, 140, 0, 0.6);
  --normal-range-fill: rgba(0, 208, 132, 0.08);
  
  --card-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  --glow-blue: 0 0 20px rgba(74, 158, 255, 0.2);
  
  --radius-sm: 6px;
  --radius-md: 12px;
  --radius-lg: 20px;
  
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 40px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
  background: var(--bg-primary); 
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* Card component */
.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: var(--spacing-lg);
  box-shadow: var(--card-shadow);
}

/* Risk tier badge */
.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.risk-badge.GREEN  { background: var(--risk-green-bg); color: var(--risk-green); }
.risk-badge.YELLOW { background: var(--risk-yellow-bg); color: var(--risk-yellow); }
.risk-badge.ORANGE { background: var(--risk-orange-bg); color: var(--risk-orange); }
.risk-badge.RED    { background: var(--risk-red-bg); color: var(--risk-red); animation: pulse-red 2s infinite; }

@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255, 61, 0, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(255, 61, 0, 0); }
}

/* Glucose value display */
.glucose-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 48px;
  font-weight: 700;
  line-height: 1;
}

/* Trend arrow */
.trend-arrow { 
  font-size: 24px;
  transition: transform 0.3s ease;
}

Add Google Fonts import at top: Inter and JetBrains Mono.
Then create src/dashboard/src/App.jsx with the main layout:
  - Sidebar (280px) + Main content area
  - Top navbar with GlucoTwin logo + system status
  - Main grid: TwinStatusBar full-width, then 2-column below
    Left (7/12): GlucoseTimeline + AlertPanel
    Right (5/12): WearableMetrics + EHRSummary + SHAPChart
```

---

## 🤖 Vibe Prompt 2 — WebSocket Hook + Patient Context

```
Create src/dashboard/src/hooks/useWebSocket.js:

import { useEffect, useRef, useState, useCallback } from 'react';

export function useWebSocket(patientId, onMessage) {
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    const wsUrl = `ws://localhost:8000/ws/patients/${patientId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      console.log(`WebSocket connected for patient ${patientId}`);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
      onMessage?.(data);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    wsRef.current = ws;

    // Keepalive ping every 25 seconds
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 25000);

    return () => clearInterval(pingInterval);
  }, [patientId, onMessage]);

  useEffect(() => {
    const cleanup = connect();
    return () => {
      cleanup?.();
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { isConnected, lastMessage };
}

Also create src/dashboard/src/hooks/usePatientTwin.js:
  - Manages all patient twin state
  - Provides: currentPatient, setCurrentPatient, patients, twinData
  - Calls REST API to load initial patient list
  - Uses useWebSocket to get real-time updates
  - Maintains glucose history array (last 24h) for chart
  - Triggers alert sound when risk_tier changes to RED
```

---

## 🤖 Vibe Prompt 3 — Glucose Timeline Chart

```
Create src/dashboard/src/components/GlucoseTimeline.jsx

This is the HERO component — the most important visual in the dashboard.

Requirements:
1. Use Recharts ComposedChart (combines Area + Line + ReferenceArea)
2. Time axis: last 4 hours of historical + 2 hours future (prediction band)
3. Y-axis: 40-400 mg/dL with clinical zone annotations

Chart layers (bottom to top):
  a. Green reference area: 70-180 mg/dL "Normal Range" (shaded)
  b. Red reference line: 250 mg/dL "Hyperglycaemia threshold"
  c. Red reference line: 70 mg/dL "Hypoglycaemia threshold"
  d. Blue area: Historical glucose (solid fill, soft blue)
  e. Orange dashed area: Prediction band (glucose_lower to glucose_upper)
  f. Orange line: Projected glucose trajectory
  g. Red vertical markers: Alert timestamps (vertical lines)
  h. Green dot: Current glucose value (animated pulse)

Custom tooltip showing:
  - Timestamp (formatted as HH:mm)
  - Glucose value + units
  - If forecast point: "Predicted" label
  - HRV at that time
  - Activity type

Props:
  glucoseHistory: Array of {timestamp, glucose_mgdl, risk_tier}
  glucoseForecast: Array of {timestamp, glucose_projected, glucose_lower, glucose_upper}
  alerts: Array of {timestamp, tier}
  onTimePointClick: callback when user clicks a data point

Styling:
  - Chart background: var(--bg-surface)
  - Grid lines: subtle (var(--border-color))
  - Smooth curves (type="monotone")
  - Animated entry (Recharts animation)
  - Responsive (ResponsiveContainer width="100%")

Add a "LIVE" indicator dot that pulses when WebSocket is connected.
Add a time range selector: 1h | 4h | 8h | 24h buttons.
```

---

## 🤖 Vibe Prompt 4 — SHAP Waterfall Chart + Alert Panel

```
Create two components:

1. src/dashboard/src/components/SHAPWaterfallChart.jsx

Render a horizontal waterfall bar chart showing SHAP feature contributions.

Data: shapFeatures = [
  { feature: "glucose_roc_current", clinical_label: "Glucose rising rapidly (+3.2 mg/dL/min)", 
    shap_value: 0.18, direction: "increases_risk" },
  { feature: "hrv_suppression_flag", clinical_label: "HRV suppression detected",
    shap_value: 0.14, direction: "increases_risk" },
  ...
]

Visual design:
- Horizontal bars, sorted by |SHAP value| descending
- Bars extending RIGHT = increases_risk → red/orange color
- Bars extending LEFT = decreases_risk → green color
- Base value shown as a vertical dashed line
- Final prediction shown at top: "Risk Score: 0.83"
- Feature labels on left, values on right (+0.18, -0.07, etc.)
- Animate bars on mount (Framer Motion if available, else CSS animation)

Show only when risk_tier === "RED". Otherwise show:
  "Explanation available when RED alert is active."

2. src/dashboard/src/components/AlertPanel.jsx

Panel showing active and recent alerts.

For each alert:
  - Timestamp (relative: "3 minutes ago")
  - Risk tier badge (ORANGE/RED with appropriate color)
  - Event type (Hyperglycaemia / Hypoglycaemia)
  - Alert message (clinical text)
  - "View Explanation" button → toggles SHAP chart

If no active alerts:
  - Show: "✅ All patients within normal range" (green)

RED alerts should have:
  - Pulsing red border animation
  - 🚨 icon
  - Sound alert (browser notification API if permitted)
  
ORANGE alerts:
  - Solid orange border
  - ⚠️ icon
```

---

## 🤖 Vibe Prompt 5 — Patient Selector + Wearable Metrics Cards

```
Create two more components:

1. src/dashboard/src/components/PatientSelector.jsx (sidebar)

A vertical list of patient cards showing:
  - Patient name (truncated)
  - Age + gender
  - Risk tier badge (color-coded)
  - Current glucose + trend arrow
  - Last updated time

Clicking a patient:
  - Sets as active patient (highlighted border)
  - Triggers WebSocket subscription change
  - Loads patient history from API

Add a search input at top to filter patients.
Add a sort dropdown: by risk (high first) | by name | by last updated.

Style: Compact cards, ~60px height, hover state with var(--brand-blue-dim) background.

2. src/dashboard/src/components/WearableMetrics.jsx

4 metric cards in a 2×2 grid:

Card 1 — Glucose (large, current value):
  - Glucose value (big, colored by risk: green/yellow/orange/red)
  - Trend arrow (↑↑ ↑ → ↓ ↓↓) with label ("Rapidly Rising")
  - "Time in Range today: 74%"

Card 2 — Heart Rate Variability:
  - RMSSD value in ms
  - "Normal: 25-50ms for age 50" (reference)
  - Sparkline of last 1h RMSSD (mini chart)
  - Color: green if above baseline, orange if suppressed

Card 3 — Activity:
  - Steps today: 4,230
  - Current activity type (walking / resting / sleeping)
  - Activity ring (circular progress) showing daily step goal
  - Last exercise: "2h ago"

Card 4 — Sleep:
  - Sleep quality: 72% (last night)
  - Sleep stages mini chart (hypnogram)
  - "Deep sleep: 1h 20min" 
  - Correlation note: "Poor sleep → elevated fasting glucose"

Also create src/dashboard/src/components/TwinStatusBar.jsx:
  Full-width top bar showing:
  - Patient name + "Digital Twin" label
  - Glucose (large): 187 mg/dL ↑
  - Risk tier badge (animated if RED)
  - IOB: 1.2 units
  - Last updated: "12 seconds ago"
  - Connection status dot (green=WS connected, red=offline)
```

---

## 🤖 Vibe Prompt 6 — EHR Summary Panel

```
Create src/dashboard/src/components/EHRSummary.jsx

Shows the patient's static clinical profile — the "static half" of the 
Digital Twin.

Layout: Compact two-column grid

Section 1 — Demographics:
  Age: 52  |  Gender: Male  |  BMI: 27.2 (Overweight)
  Ethnicity: South Asian  |  City: Mumbai

Section 2 — Diabetes Profile:
  HbA1c: 8.4% 🟡  |  Duration: 6 years
  Fasting Glucose: 158 mg/dL
  Type: Type 2 (Non-Insulin Dependent)

Section 3 — Medications (pill badges):
  🟢 Metformin 1000mg  🟢 Empagliflozin 10mg

Section 4 — Comorbidities (flag chips):
  🔴 Hypertension  (no dyslipidaemia, no neuropathy)

Section 5 — Personalized Twin Parameters:
  Glucose Baseline: 162 mg/dL (calibrated)
  Insulin Sensitivity: 50 mg/dL/unit
  Twin Calibrated: ✅ Yes (24h of data)
  Data Quality: 96%

Add a "⚠️ Research Prototype — Not for Clinical Use" disclaimer in muted text.

All values should update from the API response.
Use tooltips to explain clinical terminology (HbA1c, IOB, etc.).
```

---

## ✅ Dashboard Verification

```bash
# Start the dashboard dev server
cd src/dashboard
npm install
npm run dev
# Open http://localhost:3000

# Check:
# 1. Dark clinical theme visible
# 2. Patient sidebar shows 4 demo patients
# 3. Select "DEMO-RAJESH" → charts load
# 4. Glucose chart shows historical + prediction band
# 5. Risk badge animates when RED
# 6. WebSocket indicator: green dot (connected)
# 7. Trigger demo event: visit http://localhost:8000/api/v1/demo/inject-event?...
# 8. Watch chart update in real-time
# 9. When RED fires: SHAP chart appears in alert panel
# 10. Test on mobile viewport (375px) — should be usable
```

---

## 📋 Session 09 Completion Checklist

- [ ] React app runs at localhost:3000
- [ ] Dark clinical theme applied throughout
- [ ] Patient sidebar lists 4 demo patients with risk badges
- [ ] Glucose chart shows historical + forecast band
- [ ] WebSocket connects and updates chart in real-time
- [ ] Alert panel shows ORANGE/RED alerts
- [ ] SHAP waterfall visible for RED alerts
- [ ] Wearable metrics cards display current data
- [ ] EHR summary panel shows patient profile
- [ ] Mobile responsive (375px+ works)
- [ ] Pulsing animation on RED alerts
- [ ] Dashboard WOWs judges visually (premium dark theme)

## 🔜 Next Session

→ **10-deployment-testing.md** — Docker Compose, integration tests, demo prep

---

*GlucoTwin Vibe Assist 09 | Dashboard UI | Digital Twin Challenge 2026*
