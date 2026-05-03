import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  const modes = [
    {
      title: "Textual Analysis",
      description:
        "Answer a short clinical questionnaire covering your demographics, symptoms, and lifestyle to get an AI-powered PCOS risk assessment.",
      path: "/predict/textual",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      ),
    },
    {
      title: "Ultrasound Scan",
      description:
        "Upload an ovarian ultrasound image and let our deep-learning model (EfficientNetB0) detect PCOS patterns automatically.",
      path: "/predict/ultrasound",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
      ),
    },
    {
      title: "Combined Analysis",
      description:
        "Get the most comprehensive prediction by combining your clinical data with an ultrasound image using a weighted ensemble model.",
      path: "/predict/combined",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="16" />
          <line x1="8" y1="12" x2="16" y2="12" />
        </svg>
      ),
    },
  ];

  return (
    <div className="page">
      {/* Hero */}
      <section className="hero fade-in">
        <h1>
          <span className="hero-gradient">PCOS</span> Multimodal Predictor
        </h1>
        <p>
          An AI-powered tool that predicts Polycystic Ovary Syndrome risk using
          clinical data, ultrasound imaging, or a combination of both.
        </p>
      </section>

      {/* Mode cards */}
      <div className="mode-cards">
        {modes.map((mode) => (
          <div
            key={mode.path}
            className="card card-clickable mode-card"
            id={`mode-${mode.path.split("/").pop()}`}
            onClick={() => navigate(mode.path)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && navigate(mode.path)}
          >
            <div className="mode-card-icon">{mode.icon}</div>
            <h3>{mode.title}</h3>
            <p>{mode.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
