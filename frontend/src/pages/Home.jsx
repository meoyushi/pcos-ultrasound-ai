import { useEffect, useMemo, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useNeoPage } from "../hooks/useNeoPage";
import "./neoLanding.css";

export default function Home() {
  useNeoPage();
  const navigate = useNavigate();
  const rootRef = useRef(null);
  const dotRef = useRef(null);
  const ringRef = useRef(null);
  const auraRef = useRef(null);
  const particlesRef = useRef(null);

  const modes = useMemo(
    () => [
      {
        title: "Textual Analysis",
        description:
          "Input 14 clinical parameters including hormone levels, menstrual patterns, and metabolic markers to receive a Random Forest prediction.",
        path: "/predict/textual",
        icon: "📋",
        tag: "Random Forest",
        featured: false,
      },
      {
        title: "Ultrasound AI",
        description:
          "Upload your ovarian ultrasound image for deep analysis using a fine-tuned EfficientNetB0 model trained on curated medical imaging data.",
        path: "/predict/ultrasound",
        icon: "🔬",
        tag: "EfficientNetB0",
        featured: true,
      },
      {
        title: "Combined Mode",
        description:
          "The most comprehensive option — a 60/40 weighted ensemble combining textual and imaging signals for maximum prediction confidence.",
        path: "/predict/combined",
        icon: "⬡",
        tag: "Ensemble Model",
        featured: false,
      },
    ],
    []
  );

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    // Enable cursorless behavior only on this page
    root.classList.add("neo-cursorless");

    // ===== Cursor tracking (dot instant, ring + aura lag)
    const dot = dotRef.current;
    const ring = ringRef.current;
    const aura = auraRef.current;
    if (!dot || !ring || !aura) return;

    let mx = window.innerWidth / 2;
    let my = window.innerHeight / 2;
    let rx = mx;
    let ry = my;
    let ax = mx;
    let ay = my;
    let raf = 0;

    const onMove = (e) => {
      mx = e.clientX;
      my = e.clientY;
      dot.style.left = `${mx}px`;
      dot.style.top = `${my}px`;
    };

    const animate = () => {
      rx += (mx - rx) * 0.15;
      ry += (my - ry) * 0.15;
      ring.style.left = `${rx}px`;
      ring.style.top = `${ry}px`;

      ax += (mx - ax) * 0.06;
      ay += (my - ay) * 0.06;
      aura.style.left = `${ax}px`;
      aura.style.top = `${ay}px`;

      raf = window.requestAnimationFrame(animate);
    };

    window.addEventListener("mousemove", onMove);
    raf = window.requestAnimationFrame(animate);

    // Cursor scale on hover
    const hoverables = root.querySelectorAll("a, button, .neo-mode-card, .neo-auth-tab, .neo-nav-cta");
    const onEnter = () => {
      dot.style.width = "14px";
      dot.style.height = "14px";
      ring.style.width = "50px";
      ring.style.height = "50px";
      ring.style.borderColor = "rgba(168,197,160,0.9)";
    };
    const onLeave = () => {
      dot.style.width = "8px";
      dot.style.height = "8px";
      ring.style.width = "36px";
      ring.style.height = "36px";
      ring.style.borderColor = "rgba(168,197,160,0.6)";
    };
    hoverables.forEach((el) => {
      el.addEventListener("mouseenter", onEnter);
      el.addEventListener("mouseleave", onLeave);
    });

    // ===== Particles
    const pContainer = particlesRef.current;
    const created = [];
    if (pContainer) {
      for (let i = 0; i < 18; i += 1) {
        const p = document.createElement("div");
        p.className = "neo-particle";
        const size = Math.random() * 4 + 2;
        p.style.width = `${size}px`;
        p.style.height = `${size}px`;
        p.style.left = `${Math.random() * 100}%`;
        p.style.animationDuration = `${Math.random() * 20 + 15}s`;
        p.style.animationDelay = `-${Math.random() * 20}s`;
        p.style.opacity = `${Math.random() * 0.5 + 0.1}`;
        pContainer.appendChild(p);
        created.push(p);
      }
    }

    // ===== Scroll reveal
    const revealEls = root.querySelectorAll(".neo-reveal");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.classList.add("visible");
        });
      },
      { threshold: 0.15 }
    );
    revealEls.forEach((el) => observer.observe(el));

    // ===== 3D tilt cards
    const cards = root.querySelectorAll(".neo-mode-card");
    const onCardMove = (card) => (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `translateY(-8px) rotateX(${-y * 6}deg) rotateY(${x * 6}deg)`;
      card.style.transition = "transform 0.1s ease";
    };
    const onCardLeave = (card) => () => {
      card.style.transform = "";
      card.style.transition =
        "transform 0.4s cubic-bezier(0.23,1,0.32,1), border-color 0.3s, background 0.3s";
    };
    const cardHandlers = [];
    cards.forEach((card) => {
      const mm = onCardMove(card);
      const ml = onCardLeave(card);
      card.addEventListener("mousemove", mm);
      card.addEventListener("mouseleave", ml);
      cardHandlers.push([card, mm, ml]);
    });

    // ===== Ripple clicks
    const rippleTargets = root.querySelectorAll(".neo-btn-primary");
    const rippleHandlers = [];
    rippleTargets.forEach((btn) => {
      const onClick = (e) => {
        const r = document.createElement("span");
        r.className = "neo-ripple";
        const rect = btn.getBoundingClientRect();
        r.style.left = `${e.clientX - rect.left - 10}px`;
        r.style.top = `${e.clientY - rect.top - 10}px`;
        btn.appendChild(r);
        window.setTimeout(() => r.remove(), 700);
      };
      btn.addEventListener("click", onClick);
      rippleHandlers.push([btn, onClick]);
    });

    return () => {
      root.classList.remove("neo-cursorless");
      window.removeEventListener("mousemove", onMove);
      window.cancelAnimationFrame(raf);
      hoverables.forEach((el) => {
        el.removeEventListener("mouseenter", onEnter);
        el.removeEventListener("mouseleave", onLeave);
      });
      created.forEach((p) => p.remove());
      observer.disconnect();
      cardHandlers.forEach(([card, mm, ml]) => {
        card.removeEventListener("mousemove", mm);
        card.removeEventListener("mouseleave", ml);
      });
      rippleHandlers.forEach(([btn, onClick]) => btn.removeEventListener("click", onClick));
    };
  }, []);

  return (
    <div className="neo" ref={rootRef}>
      <div className="neo-cursor-dot" ref={dotRef} />
      <div className="neo-cursor-ring" ref={ringRef} />
      <div className="neo-cursor-aura" ref={auraRef} />

      <div className="neo-orb-container" aria-hidden="true">
        <div className="neo-orb neo-orb-1" />
        <div className="neo-orb neo-orb-2" />
        <div className="neo-orb neo-orb-3" />
        <div className="neo-orb neo-orb-4" />
      </div>
      <div className="neo-grid-lines" aria-hidden="true" />
      <div className="neo-particles" ref={particlesRef} aria-hidden="true" />

      <nav className="neo-nav">
        <div className="neo-nav-logo" onClick={() => navigate("/")} role="button" tabIndex={0}>
          ⬡ <span>PCOS</span>AI
        </div>
        <ul className="neo-nav-links">
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/predict/combined">Predict</Link>
          </li>
          <li>
            <Link to="/predict/textual">Textual</Link>
          </li>
          <li>
            <Link to="/predict/ultrasound">Ultrasound</Link>
          </li>
          <li>
            <Link className="neo-nav-cta" to="/auth?tab=login">
              Sign In
            </Link>
          </li>
        </ul>
      </nav>

      <section className="neo-hero">
        <div className="neo-badge neo-badge-1">
          <span className="neo-badge-dot" />
          AI Model Active
        </div>
        <div className="neo-badge neo-badge-2">✦ EfficientNetB0 + Random Forest</div>
        <div className="neo-badge neo-badge-3">◎ 3 Prediction Modes</div>

        <div className="neo-hero-inner">
          <div className="neo-eyebrow">AI-Powered Women&apos;s Health</div>
          <h1 className="neo-hero-title">
            <span className="neo-hero-text">Understand your</span>
            <br />
            <em>hormonal</em> <span className="neo-hero-text">health</span>
            <br />
            <span className="neo-hero-text">with </span>
            <strong>precision</strong>
          </h1>
          <p className="neo-hero-sub">
            Advanced multimodal AI for PCOS risk assessment — combining clinical data, ultrasound imaging, and ensemble
            prediction for actionable insights.
          </p>
          <div className="neo-ctas">
            <Link className="neo-btn-primary" to="/auth?tab=signup">
              Begin Assessment →
            </Link>
            <a className="neo-btn-ghost" href="#modes">
              Explore Methods
            </a>
          </div>
        </div>

        <div className="neo-scroll-hint" aria-hidden="true">
          Scroll
          <div className="neo-scroll-line" />
        </div>
      </section>

      <div className="neo-section" style={{ paddingTop: 0 }}>
        <div className="neo-stats-row neo-reveal">
          <div className="neo-stat-item">
            <div className="neo-stat-num">
              94<span style={{ fontSize: "1.5rem" }}>%</span>
            </div>
            <div className="neo-stat-label">Accuracy (EfficientNetB0)</div>
          </div>
          <div className="neo-stat-item">
            <div className="neo-stat-num">14</div>
            <div className="neo-stat-label">Clinical Parameters</div>
          </div>
          <div className="neo-stat-item">
            <div className="neo-stat-num">3</div>
            <div className="neo-stat-label">Prediction Modalities</div>
          </div>
        </div>
      </div>

      <section id="modes" className="neo-section" style={{ paddingTop: "2rem" }}>
        <div className="neo-reveal">
          <div className="neo-section-label">How It Works</div>
          <h2 className="neo-section-title">
            Three ways to <em>assess</em>
            <br />
            your health
          </h2>
          <p className="neo-section-sub">
            Choose the method that fits your situation. Each mode uses a different AI model optimised for that data type.
          </p>
        </div>

        <div className="neo-modes-grid">
          {modes.map((m) => (
            <div
              key={m.path}
              className={`neo-mode-card neo-reveal ${m.featured ? "featured" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => navigate(m.path)}
              onKeyDown={(e) => e.key === "Enter" && navigate(m.path)}
            >
              <div className="neo-mode-icon">{m.icon}</div>
              <h3>{m.title}</h3>
              <p>{m.description}</p>
              <span className="neo-mode-tag">{m.tag}</span>
            </div>
          ))}
        </div>
      </section>

      <div className="neo-cta-section">
        <div className="neo-cta-inner neo-reveal">
          <h2>
            Ready to <em>understand</em> more?
          </h2>
          <p>
            Create a free account to save your assessments, track changes over time, and share results with your
            healthcare provider.
          </p>
          <div className="neo-ctas">
            <Link className="neo-btn-primary" to="/auth?tab=signup">
              Create Free Account
            </Link>
            <Link className="neo-btn-ghost" to="/predict/combined">
              Learn About PCOS
            </Link>
          </div>
        </div>
      </div>

      <footer className="neo-footer">
        <span>⬡ PCOS Multimodal Predictor — Not a substitute for medical advice</span>
        <span>Built with EfficientNetB0 · Random Forest · React 18</span>
      </footer>
    </div>
  );
}
