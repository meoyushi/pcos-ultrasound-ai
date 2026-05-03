import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SymptomForm from "../components/SymptomForm";
import { getDefaultValues } from "../utils/fields";

const API = import.meta.env.VITE_API_URL || "";

export default function TextualPredict() {
  const navigate = useNavigate();
  const [values, setValues] = useState(getDefaultValues());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (name, value) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    // Convert string values to numbers for the API
    const features = {};
    Object.entries(values).forEach(([key, val]) => {
      features[key] = parseFloat(val) || 0;
    });

    try {
      const res = await fetch(`${API}/api/predict/textual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Server error (${res.status})`);
      }

      const result = await res.json();
      navigate("/result", { state: result });
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page fade-in">
      {/* Loading overlay */}
      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <p className="spinner-text">Analyzing your data…</p>
        </div>
      )}

      {/* Back button */}
      <button className="back-btn" onClick={() => navigate("/")} type="button">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="19" y1="12" x2="5" y2="12" />
          <polyline points="12 19 5 12 12 5" />
        </svg>
        Back to Home
      </button>

      <div className="page-header">
        <h1>Textual Analysis</h1>
        <p>Fill in your clinical details below for an AI-powered PCOS risk assessment.</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="card">
          <SymptomForm values={values} onChange={handleChange} />
        </div>

        {error && <div className="error-message">{error}</div>}

        <div style={{ textAlign: "center", marginTop: 28 }}>
          <button type="submit" className="btn-primary" disabled={loading}>
            Predict PCOS Risk
          </button>
        </div>
      </form>
    </div>
  );
}
