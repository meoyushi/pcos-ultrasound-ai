import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";

const API = import.meta.env.VITE_API_URL || "";
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

export default function UltrasoundPredict() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (f) => {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setError("Please upload a JPG, PNG, or WebP image.");
      return;
    }
    setError("");
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const removeFile = () => {
    setFile(null);
    setPreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an ultrasound image.");
      return;
    }

    setError("");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/api/predict/ultrasound`, {
        method: "POST",
        body: formData,
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
      {loading && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <p className="spinner-text">Analyzing ultrasound…</p>
        </div>
      )}

      <button className="back-btn" onClick={() => navigate("/")} type="button">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="19" y1="12" x2="5" y2="12" />
          <polyline points="12 19 5 12 12 5" />
        </svg>
        Back to Home
      </button>

      <div className="page-header">
        <h1>Ultrasound Scan Analysis</h1>
        <p>Upload an ovarian ultrasound image for AI-powered PCOS detection.</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="card">
          {/* Drop zone */}
          <div
            className={`drop-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
          >
            <div className="drop-zone-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: "100%", height: "100%" }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="drop-zone-text">
              {file ? "Click or drag to replace" : "Drag & drop your ultrasound image here"}
            </p>
            <p className="drop-zone-hint">or click to browse · JPG, PNG, WebP</p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => handleFile(e.target.files[0])}
            style={{ display: "none" }}
          />

          {/* Preview */}
          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Ultrasound preview" />
              <p className="file-name">{file.name}</p>
              <button type="button" className="remove-btn" onClick={removeFile}>
                Remove image
              </button>
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        <div style={{ textAlign: "center", marginTop: 28 }}>
          <button type="submit" className="btn-primary" disabled={loading || !file}>
            Analyze Ultrasound
          </button>
        </div>
      </form>
    </div>
  );
}
