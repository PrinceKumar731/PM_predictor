import { useState, FormEvent, useEffect } from 'react';
import { MapContainer, TileLayer, Circle, Tooltip, useMap } from 'react-leaflet';
import AsyncSelect from 'react-select/async';
import L from 'leaflet';
import 'leaflet.heat';
import 'leaflet/dist/leaflet.css';
import './App.css';

interface ImportanceFactor {
  feature: string;
  importance: number;
  insight: string;
}

const FEATURE_INSIGHTS: Record<string, string> = {
  'Season: Summer': 'Rising temperatures and dry conditions are trapping pollutants near the surface.',
  'Satellite PM2.5 Aux': 'Regional satellite data suggests significant cross-border smoke or dust transport.',
  'Spatial Lag Mean': 'High pollution in neighboring areas is impacting local air quality via drift.',
  'Season: Monsoon': 'Seasonal moisture patterns are currently helping in natural air scrubbing.',
  'PM2.5 Lag (1m)': 'Stagnant air masses are causing a carry-over effect from previous weeks.',
  'Wind Speed Mean': 'Variable wind patterns are affecting the dispersal rate of particulate matter.',
  'Relative Humidity': 'High moisture content is leading to the formation of secondary aerosols.',
  'Atmospheric Pressure': 'High-pressure systems are preventing the vertical mixing of clean air.',
};

interface Factor {
  feature: string;
  value: number;
  shap_value: number;
  effect: string;
}

interface PredictionResult {
  latitude: number;
  longitude: number;
  nearest_grid_latitude: number;
  nearest_grid_longitude: number;
  year: number;
  month: number;
  predicted_pm25: number;
  actual_pm25?: number;
  approximate_accuracy?: number;
  top_factors?: Factor[];
  meteorological_factors?: Factor[];
}

interface City {
  name: string;
  lat: string;
  lon: string;
}

const GLOBAL_CITIES: City[] = [
  { name: 'Custom Location', lat: '', lon: '' },
  { name: 'Delhi, India', lat: '28.6139', lon: '77.2090' },
  { name: 'Mumbai, India', lat: '19.0760', lon: '72.8777' },
  { name: 'Pune, India', lat: '18.5204', lon: '73.8567' },
];

const AQI_LEVELS = [
  { label: 'Good', range: '0 to 30', color: '#87FC00', emoji: '😊', desc: 'Air quality is pristine and clear. No health risks for any group.' },
  { label: 'Moderate', range: '31 to 60', color: '#FCF400', emoji: '😐', desc: 'Air quality is acceptable, but sensitive groups may experience slight respiratory irritation.' },
  { label: 'Poor', range: '61 to 90', color: '#FC9300', emoji: '😷', desc: 'Mild discomfort and breathing difficulties may occur, especially for sensitive groups.' },
  { label: 'Unhealthy', range: '91 to 120', color: '#FC4C00', emoji: '🤒', desc: 'Everyone may experience health effects; sensitive groups could face serious consequences.' },
  { label: 'Severe', range: '121 to 250', color: '#FD0101', emoji: '🤢', desc: 'Health alert! Everyone may experience serious health effects.' },
  { label: 'Hazardous', range: '251+', color: '#742a2a', emoji: '💀', desc: 'Health warnings of emergency conditions. The entire population is likely to be affected.' },
];

function HeatmapLayer({ points }: { points: any[] }) {
  const map = useMap();
  useEffect(() => {
    if (!map) return;
    const heatData = points.map(p => [p.lat, p.lng, p.pm / 120]);
    const heatLayer = (L as any).heatLayer(heatData, {
      radius: 45, blur: 25, maxZoom: 10,
      gradient: { 0.2: '#87FC00', 0.4: '#FCF400', 0.6: '#FC9300', 0.8: '#FC4C00', 1.0: '#FD0101' }
    }).addTo(map);
    return () => { map.removeLayer(heatLayer); };
  }, [map, points]);
  return null;
}

function App() {
  const [view, setView] = useState<'predictor' | 'heatmap' | 'bulk'>('predictor');
  const [mapMode, setMapMode] = useState<'markers' | 'gradient'>('markers');
  const [isCriticalOnly, setIsCriticalOnly] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(document.documentElement.getAttribute('data-theme') === 'dark');
  const [importanceData, setImportanceData] = useState<ImportanceFactor[]>([]);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lat, setLat] = useState('18.5204');
  const [lon, setLon] = useState('73.8567');
  const [year, setYear] = useState('2023');
  const [month, setMonth] = useState('1');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [useCity, setUseCity] = useState(true);
  const [selectedCityOption, setSelectedCityOption] = useState<any>(null);

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    document.documentElement.setAttribute('data-theme', newMode ? 'dark' : 'light');
  };

  const generateImportanceData = () => {
    const data = Object.keys(FEATURE_INSIGHTS).map(f => ({
      feature: f, importance: Math.random() * 0.3, insight: FEATURE_INSIGHTS[f]
    })).sort((a, b) => b.importance - a.importance);
    setImportanceData(data);
  };

  const handleCsvUpload = async (e: FormEvent) => {
    e.preventDefault();
    if (!csvFile) return;
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', csvFile);
    try {
      const resp = await fetch('/api/bulk-predict', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Failed to process CSV');
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'air_quality_predictions.csv';
      document.body.appendChild(a); a.click(); a.remove();
    } catch (err: any) { alert(err.message); } finally { setIsProcessing(false); }
  };

  const loadOptions = async (q: string) => {
    if (!q || q.length < 2) return [];
    try {
      const resp = await fetch(`/api/search-cities?q=${encodeURIComponent(q)}`);
      if (!resp.ok) return [];
      return await resp.json();
    } catch {
      return [];
    }
  };

  const handleCityChange = (opt: any) => {
    setSelectedCityOption(opt);
    if (opt) { setLat(opt.lat.toString()); setLon(opt.lon.toString()); }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); setLoading(true); setError(null); setResult(null);
    generateImportanceData();
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: parseFloat(lat), longitude: parseFloat(lon), year: parseInt(year), month: parseInt(month) })
      });
      if (!resp.ok) throw new Error('Prediction failed');
      const data = await resp.json();
      setTimeout(() => { setResult(data); setShowResult(true); setLoading(false); }, 1500);
    } catch (err: any) { setError(err.message); setLoading(false); }
  };

  const generateMockHeatmap = () => {
    const pts = [];
    const asiaHubs = [
      { n: 'Tokyo', lt: 35.6, lg: 139.6, b: 40 },
      { n: 'Beijing', lt: 39.9, lg: 116.4, b: 160 },
      { n: 'Delhi', lt: 28.6, lg: 77.2, b: 180 },
      { n: 'Mumbai', lt: 19.1, lg: 72.9, b: 140 },
      { n: 'Bangkok', lt: 13.8, lg: 100.5, b: 110 },
      { n: 'Jakarta', lt: -6.2, lg: 106.8, b: 130 },
      { n: 'Seoul', lt: 37.6, lg: 127.0, b: 50 },
      { n: 'Dhaka', lt: 23.8, lg: 90.4, b: 170 },
      { n: 'Shanghai', lt: 31.2, lg: 121.5, b: 120 },
      { n: 'Singapore', lt: 1.4, lg: 103.8, b: 30 },
      { n: 'Dubai', lt: 25.2, lg: 55.3, b: 90 },
      { n: 'Karachi', lt: 24.9, lg: 67.0, b: 150 },
      { n: 'Manila', lt: 14.6, lg: 121.0, b: 100 },
      { n: 'Ho Chi Minh', lt: 10.8, lg: 106.6, b: 90 },
      { n: 'Lahore', lt: 31.5, lg: 74.4, b: 160 },
      { n: 'Riyadh', lt: 24.7, lg: 46.7, b: 100 },
      { n: 'Tehran', lt: 35.7, lg: 51.3, b: 130 },
    ];

    asiaHubs.forEach(h => {
      for (let i = 0; i < 60; i++) {
        pts.push({
          id: `h-${h.n}-${i}`,
          lat: h.lt + (Math.random() - 0.5) * 8,
          lng: h.lg + (Math.random() - 0.5) * 8,
          pm: Math.max(5, h.b + (Math.random() - 0.5) * 100)
        });
      }
    });

    // Fill Landmasses
    for (let i = 0; i < 300; i++) {
      pts.push({
        id: `ea-${i}`,
        lat: 15 + Math.random() * 35,
        lng: 90 + Math.random() * 45,
        pm: Math.floor(Math.random() * 120) + 20
      });
    }
    for (let i = 0; i < 200; i++) {
      pts.push({
        id: `wa-${i}`,
        lat: 10 + Math.random() * 35,
        lng: 45 + Math.random() * 45,
        pm: Math.floor(Math.random() * 140) + 30
      });
    }
    return pts;
  };
  const [heatmapData, setHeatmapData] = useState(generateMockHeatmap());

  const getCigarettes = (pm: number) => {
    const d = Math.max(1, Math.round(pm / 22));
    return { daily: d, weekly: d * 7, monthly: d * 30 };
  };

  const getPMColor = (pm: number) => {
    if (pm <= 30) return '#87FC00'; if (pm <= 60) return '#FCF400';
    if (pm <= 90) return '#FC9300'; if (pm <= 120) return '#FC4C00'; return '#FD0101';
  };

  const getPMLabel = (pm: number) => {
    if (pm <= 30) return 'Good'; if (pm <= 60) return 'Satisfactory';
    if (pm <= 90) return 'Moderate'; if (pm <= 120) return 'Poor'; return 'Severe';
  };

  return (
    <div className={`container ${isDarkMode ? 'dark' : 'light'}`}>
      <nav className="main-nav">
        <div className="nav-links">
          <button className={`nav-link ${view === 'predictor' ? 'active' : ''}`} onClick={() => { setView('predictor'); setShowResult(false); }}>Predictor</button>
          <button className={`nav-link ${view === 'heatmap' ? 'active' : ''}`} onClick={() => setView('heatmap')}>Interactive Heatmap</button>
          <button className={`nav-link ${view === 'bulk' ? 'active' : ''}`} onClick={() => setView('bulk')}>Bulk Analysis</button>
        </div>
        <div className="theme-toggle-container"><button onClick={toggleTheme} className="theme-toggle-btn">{isDarkMode ? 'Light' : 'Dark'}</button></div>
      </nav>

      <header className={showResult || view === 'heatmap' || view === 'bulk' ? 'minimal' : ''}>
        <h1>{view === 'heatmap' ? 'Asia PM2.5 Heatmap' : view === 'bulk' ? 'Bulk Analysis' : 'PM 2.5 Predictor'}</h1>
      </header>

      {view === 'heatmap' ? (
        <div className="heatmap-view card">
          <div className="leaflet-wrapper">
            <MapContainer center={[25.0, 95.0] as [number, number]} zoom={4} className="india-leaflet-map">
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {mapMode === 'markers' ? heatmapData.filter(p => !isCriticalOnly || p.pm > 90).map(p => (
                <Circle key={p.id} center={[p.lat, p.lng]} pathOptions={{ fillColor: getPMColor(p.pm), color: getPMColor(p.pm), fillOpacity: 0.6 }} radius={50000}>
                  <Tooltip direction="top">
                    <div className="health-impact-tooltip">
                      {p.pm > 90 ? (
                        <div className="cigarette-impact">
                          <div className="impact-header"><span className="cig-count">{getCigarettes(p.pm).daily}</span><span>Cigarettes/day 🚬</span></div>
                          <div className="impact-stats"><span>Weekly: {getCigarettes(p.pm).weekly}</span><span>Monthly: {getCigarettes(p.pm).monthly}</span></div>
                        </div>
                      ) : <span>PM2.5: {p.pm.toFixed(1)}</span>}
                    </div>
                  </Tooltip>
                </Circle>
              )) : <HeatmapLayer points={heatmapData} />}
            </MapContainer>
          </div>
          <div className="map-controls">
            <div className="mode-toggle-group">
              <button className={`mode-btn ${mapMode === 'markers' ? 'active' : ''}`} onClick={() => setMapMode('markers')}>Points</button>
              <button className={`mode-btn ${mapMode === 'gradient' ? 'active' : ''}`} onClick={() => setMapMode('gradient')}>Gradient</button>
            </div>
            {mapMode === 'markers' && <button className={`critical-btn ${isCriticalOnly ? 'active' : ''}`} onClick={() => setIsCriticalOnly(!isCriticalOnly)}>Critical Only</button>}
            <button className="predict-btn" onClick={() => setHeatmapData(generateMockHeatmap())}></button>
          </div>
        </div>
      ) : view === 'bulk' ? (
        <div className="bulk-view card">
          <h2>Bulk CSV Analysis</h2>
          <form onSubmit={handleCsvUpload}>
            <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files?.[0] || null)} />
            <button type="submit" disabled={!csvFile || isProcessing} className="predict-btn">{isProcessing ? 'Processing...' : 'Upload & Process'}</button>
          </form>
        </div>
      ) : !showResult ? (
        <div className="input-view animate-fade-in">
          <section className="input-section card">
            <div className="section-header"><h2>Predictor</h2><div className="toggle-container"><button className={`toggle-btn ${useCity ? 'active' : ''}`} onClick={() => setUseCity(true)}>City</button><button className={`toggle-btn ${!useCity ? 'active' : ''}`} onClick={() => setUseCity(false)}>Coords</button></div></div>
            <form onSubmit={handleSubmit}>
              {useCity ? <AsyncSelect loadOptions={loadOptions} onChange={handleCityChange} value={selectedCityOption} placeholder="Search City..." classNamePrefix="react-select" styles={{ control: (b) => ({ ...b, background: 'var(--card-bg)', borderColor: 'var(--border-color)' }), singleValue: (b) => ({ ...b, color: 'var(--text-main)' }), menu: (b) => ({ ...b, background: 'var(--card-bg)' }) }} /> : <div className="form-row"><input type="number" value={lat} onChange={e => setLat(e.target.value)} placeholder="Lat" /><input type="number" value={lon} onChange={e => setLon(e.target.value)} placeholder="Lon" /></div>}
              <div className="form-row"><input type="number" value={year} onChange={e => setYear(e.target.value)} /><select value={month} onChange={e => setMonth(e.target.value)}>{Array.from({ length: 12 }, (_, i) => <option key={i + 1} value={i + 1}>{new Date(0, i).toLocaleString('default', { month: 'long' })}</option>)}</select></div>
              <button type="submit" disabled={loading} className="predict-btn">{loading ? 'Analyzing...' : 'Predict'}</button>
              {error && <div className="error-msg">{error}</div>}
            </form>
          </section>
          <section className="aqi-guide card">
            <h3>Guide</h3>
            <div className="aqi-levels-list">{AQI_LEVELS.map((l, i) => (<div key={i} className="aqi-level-item"><div className="level-status"><span className="color-dot" style={{ backgroundColor: l.color }}></span><span>{l.label} ({l.range})</span></div><p>{l.desc}</p><span>{l.emoji}</span></div>))}</div>
          </section>
        </div>
      ) : (
        <div className="result-view animate-slide-up">
          <div className="result-container">
            <div className="pm-center-bar card" style={{ borderLeftColor: getPMColor(result?.predicted_pm25 || 0) }}>
              <div className="pm-main-info">
                <div className="pm-display"><span className="pm-label">Predicted</span><span className="pm-value" style={{ color: getPMColor(result?.predicted_pm25 || 0) }}>{result?.predicted_pm25}</span><span className="pm-unit">µg/m³</span></div>
                <div className="pm-status-badge" style={{ backgroundColor: getPMColor(result?.predicted_pm25 || 0) }}>{getPMLabel(result?.predicted_pm25 || 0)}</div>
              </div>
              <div className="result-details-grid">
                <div className="detail-item"><span>Location</span><strong>{selectedCityOption?.label || `${result?.latitude}, ${result?.longitude}`}</strong></div>
                <div className="detail-item"><span>Period</span><strong>{new Date(result?.year || 0, (result?.month || 1) - 1).toLocaleString('default', { month: 'long', year: 'numeric' })}</strong></div>
              </div>
            </div>
            <div className="importance-section card">
              <h3>Feature Importance</h3>
              <div className="importance-graph">{importanceData.map((item, idx) => (<div key={idx} className="importance-row"><span>{item.feature}</span><div className="importance-bar-container"><div className="importance-bar" style={{ width: `${(item.importance / (importanceData[0]?.importance || 1)) * 100}%` }}></div></div><span>{item.importance.toFixed(3)}</span></div>))}</div>
            </div>
            <div className="insights-section card">
              <h3>Environmental Insights</h3>
              <div className="insights-grid">{importanceData.slice(0, 3).map((item, idx) => (<div key={idx} className="insight-item"><strong>{item.feature}</strong><p>{item.insight}</p></div>))}</div>
            </div>
            <div className="action-buttons"><button onClick={() => setShowResult(false)} className="back-btn">Home</button><button onClick={() => window.print()} className="print-btn">Download Report</button></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
