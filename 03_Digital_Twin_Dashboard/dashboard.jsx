import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Activity, Wind, Droplets, Factory, Heart, AlertTriangle, TrendingDown, Recycle, Building2, Gauge } from 'lucide-react';

// Simulated Digital Twin - generates realistic industrial data
const generateRealtimeData = (prevData) => {
  const baseProduction = 850 + Math.sin(Date.now() / 50000) * 200;
  const so2Base = 420 + Math.random() * 80;
  const captureEfficiency = 0.87 + Math.random() * 0.08;
  
  return {
    timestamp: new Date().toLocaleTimeString(),
    so2_input: Math.round(so2Base),
    so2_captured: Math.round(so2Base * captureEfficiency),
    so2_released: Math.round(so2Base * (1 - captureEfficiency)),
    capture_efficiency: Math.round(captureEfficiency * 100),
    gypsum_processed: Math.round(baseProduction + Math.random() * 100),
    bricks_produced: Math.round((baseProduction + Math.random() * 100) * 0.42),
    temperature: Math.round(45 + Math.random() * 15),
    ph_level: (6.2 + Math.random() * 0.6).toFixed(1),
    pm25: Math.round(35 + Math.random() * 25),
    pm10: Math.round(65 + Math.random() * 35),
    health_risk_index: Math.round(25 + Math.random() * 30),
    co2_offset: Math.round(1250 + Math.random() * 200),
  };
};

// ML Prediction Simulation (represents LSTM/XGBoost output)
const generatePredictions = () => {
  const predictions = [];
  let baseValue = 380;
  for (let i = 0; i < 24; i++) {
    baseValue += (Math.random() - 0.5) * 40;
    predictions.push({
      hour: `${i}:00`,
      predicted_so2: Math.max(300, Math.min(550, Math.round(baseValue))),
      confidence_upper: Math.round(baseValue * 1.15),
      confidence_lower: Math.round(baseValue * 0.85),
    });
  }
  return predictions;
};

const generateHealthData = () => [
  { category: 'Respiratory', cases_before: 847, cases_after: 312, reduction: 63 },
  { category: 'Cardiovascular', cases_before: 423, cases_after: 198, reduction: 53 },
  { category: 'Skin Conditions', cases_before: 256, cases_after: 89, reduction: 65 },
  { category: 'Eye Irritation', cases_before: 634, cases_after: 201, reduction: 68 },
];

const COLORS = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b'];

export default function Dashboard() {
  const [realtimeData, setRealtimeData] = useState([]);
  const [currentData, setCurrentData] = useState(generateRealtimeData(null));
  const [predictions, setPredictions] = useState(generatePredictions());
  const [healthData] = useState(generateHealthData());
  const [totalStats, setTotalStats] = useState({
    so2_captured_total: 12847,
    bricks_produced_total: 45230,
    co2_offset_total: 28450,
    waste_recycled: 892,
  });

  // Simulate real-time data updates (Digital Twin)
  useEffect(() => {
    const interval = setInterval(() => {
      const newData = generateRealtimeData(currentData);
      setCurrentData(newData);
      setRealtimeData(prev => [...prev.slice(-20), newData]);
      setTotalStats(prev => ({
        so2_captured_total: prev.so2_captured_total + Math.round(newData.so2_captured / 60),
        bricks_produced_total: prev.bricks_produced_total + Math.round(newData.bricks_produced / 120),
        co2_offset_total: prev.co2_offset_total + Math.round(newData.co2_offset / 100),
        waste_recycled: prev.waste_recycled + Math.round(newData.gypsum_processed / 2000),
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, [currentData]);

  // Refresh predictions periodically
  useEffect(() => {
    const interval = setInterval(() => setPredictions(generatePredictions()), 30000);
    return () => clearInterval(interval);
  }, []);

  const StatCard = ({ icon: Icon, title, value, unit, subtext, color, trend }) => (
    <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-100">
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg ${color}`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        {trend && (
          <span className={`text-xs font-medium px-2 py-1 rounded-full ${trend > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
        )}
      </div>
      <p className="text-2xl font-bold text-gray-800">{value.toLocaleString()}<span className="text-sm font-normal text-gray-500 ml-1">{unit}</span></p>
      <p className="text-xs text-gray-500 mt-1">{title}</p>
      {subtext && <p className="text-xs text-gray-400">{subtext}</p>}
    </div>
  );

  const AlertBanner = ({ level, message }) => {
    const colors = {
      good: 'bg-green-50 border-green-200 text-green-800',
      moderate: 'bg-yellow-50 border-yellow-200 text-yellow-800',
      poor: 'bg-red-50 border-red-200 text-red-800',
    };
    return (
      <div className={`${colors[level]} border rounded-lg p-3 flex items-center gap-2`}>
        {level === 'good' ? <Activity className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
        <span className="text-sm font-medium">{message}</span>
      </div>
    );
  };

  const getHealthStatus = (index) => {
    if (index < 30) return { status: 'Good', color: 'text-green-500', bg: 'bg-green-500' };
    if (index < 50) return { status: 'Moderate', color: 'text-yellow-500', bg: 'bg-yellow-500' };
    return { status: 'Poor', color: 'text-red-500', bg: 'bg-red-500' };
  };

  const healthStatus = getHealthStatus(currentData.health_risk_index);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Factory className="w-7 h-7 text-emerald-400" />
              GCT Gabès - Digital Twin Dashboard
            </h1>
            <p className="text-slate-400 text-sm">Phosphogypsum → Bricks | Real-time Process Monitoring</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 text-emerald-400 text-sm">
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
              Live
            </span>
            <span className="text-slate-400 text-sm">{new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Alert Banner */}
      <div className="mb-4">
        <AlertBanner 
          level={currentData.capture_efficiency > 90 ? 'good' : currentData.capture_efficiency > 80 ? 'moderate' : 'poor'}
          message={`SO₂ Capture Efficiency: ${currentData.capture_efficiency}% | System ${currentData.capture_efficiency > 85 ? 'Operating Optimally' : 'Needs Attention'}`}
        />
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard icon={Wind} title="SO₂ Captured Today" value={totalStats.so2_captured_total} unit="kg" color="bg-blue-500" trend={12} />
        <StatCard icon={Building2} title="Bricks Produced" value={totalStats.bricks_produced_total} unit="units" color="bg-emerald-500" trend={8} />
        <StatCard icon={TrendingDown} title="CO₂ Offset" value={totalStats.co2_offset_total} unit="kg" color="bg-purple-500" trend={15} />
        <StatCard icon={Recycle} title="Waste Recycled" value={totalStats.waste_recycled} unit="tons" color="bg-orange-500" trend={23} />
      </div>

      {/* Real-time Process Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* SO2 Capture Chart */}
        <div className="lg:col-span-2 bg-white rounded-xl p-4 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <Wind className="w-4 h-4 text-blue-500" />
            Real-time SO₂ Monitoring (ppm)
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={realtimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Area type="monotone" dataKey="so2_input" stroke="#ef4444" fill="#fecaca" name="SO₂ Input" />
              <Area type="monotone" dataKey="so2_captured" stroke="#22c55e" fill="#bbf7d0" name="SO₂ Captured" />
              <Area type="monotone" dataKey="so2_released" stroke="#f59e0b" fill="#fef3c7" name="SO₂ Released" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Current Process Status */}
        <div className="bg-white rounded-xl p-4 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <Gauge className="w-4 h-4 text-purple-500" />
            Process Parameters
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Capture Efficiency</span>
              <span className="text-sm font-bold text-emerald-600">{currentData.capture_efficiency}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-emerald-500 h-2 rounded-full transition-all" style={{ width: `${currentData.capture_efficiency}%` }}></div>
            </div>
            
            <div className="flex justify-between items-center pt-2">
              <span className="text-xs text-gray-500">Temperature</span>
              <span className="text-sm font-semibold">{currentData.temperature}°C</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">pH Level</span>
              <span className="text-sm font-semibold">{currentData.ph_level}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Gypsum Flow</span>
              <span className="text-sm font-semibold">{currentData.gypsum_processed} kg/h</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Brick Output</span>
              <span className="text-sm font-semibold">{currentData.bricks_produced} units/h</span>
            </div>
          </div>
        </div>
      </div>

      {/* ML Predictions & Health Impact */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {/* ML Forecast */}
        <div className="bg-white rounded-xl p-4 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-700 mb-1 flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-500" />
            ML Pollution Forecast (24h) - LSTM Model
          </h3>
          <p className="text-xs text-gray-400 mb-3">Predicted SO₂ levels with 95% confidence interval</p>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={predictions}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="hour" tick={{ fontSize: 9 }} interval={3} />
              <YAxis tick={{ fontSize: 10 }} domain={[200, 600]} />
              <Tooltip />
              <Area type="monotone" dataKey="confidence_upper" stroke="transparent" fill="#c7d2fe" name="Upper Bound" />
              <Area type="monotone" dataKey="confidence_lower" stroke="transparent" fill="#ffffff" name="Lower Bound" />
              <Line type="monotone" dataKey="predicted_so2" stroke="#6366f1" strokeWidth={2} dot={false} name="Predicted SO₂" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Health Impact */}
        <div className="bg-white rounded-xl p-4 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <Heart className="w-4 h-4 text-red-500" />
            Health Impact Analysis - Gabès Region
          </h3>
          <div className="flex items-center gap-4 mb-3">
            <div className={`w-16 h-16 rounded-full ${healthStatus.bg} flex items-center justify-center`}>
              <span className="text-white text-lg font-bold">{currentData.health_risk_index}</span>
            </div>
            <div>
              <p className={`text-lg font-bold ${healthStatus.color}`}>{healthStatus.status}</p>
              <p className="text-xs text-gray-500">Current Health Risk Index</p>
              <p className="text-xs text-gray-400">PM2.5: {currentData.pm25} | PM10: {currentData.pm10}</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={healthData} layout="vertical">
              <XAxis type="number" tick={{ fontSize: 9 }} />
              <YAxis dataKey="category" type="category" tick={{ fontSize: 9 }} width={80} />
              <Tooltip />
              <Bar dataKey="cases_before" fill="#fca5a5" name="Before (cases/year)" />
              <Bar dataKey="cases_after" fill="#86efac" name="After (projected)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Environmental Impact Summary */}
      <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-4 shadow-lg text-white">
        <h3 className="text-sm font-semibold mb-3">🌍 Environmental Impact Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold">{(totalStats.so2_captured_total / 1000).toFixed(1)}t</p>
            <p className="text-xs opacity-80">SO₂ Prevented from Release</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{(totalStats.waste_recycled).toLocaleString()}t</p>
            <p className="text-xs opacity-80">Phosphogypsum Recycled</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{(totalStats.co2_offset_total / 1000).toFixed(1)}t</p>
            <p className="text-xs opacity-80">CO₂ Equivalent Offset</p>
          </div>
          <div>
            <p className="text-2xl font-bold">63%</p>
            <p className="text-xs opacity-80">Respiratory Cases Reduced</p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 text-center text-slate-500 text-xs">
        <p>🔬 ML Models: LSTM (Pollution Forecast) | XGBoost (Health Risk) | Digital Twin Simulation</p>
        <p>Hackathon Demo - GCT Gabès Circular Economy Solution</p>
      </div>
    </div>
  );
}