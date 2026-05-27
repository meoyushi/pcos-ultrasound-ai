import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import "./neoLanding.css";

export default function Auth() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const initialTab = useMemo(() => {
    const t = params.get("tab");
    return t === "signup" ? "signup" : "login";
  }, [params]);

  const [tab, setTab] = useState(initialTab);
  const [signupPass, setSignupPass] = useState("");

  const strength = useMemo(() => {
    const val = signupPass;
    const strong = val.length > 12 && /[A-Z]/.test(val) && /[0-9]/.test(val);
    if (strong) return 3;
    if (val.length > 7) return 2;
    if (val.length > 3) return 1;
    return 0;
  }, [signupPass]);

  const setTabAndUrl = (next) => {
    setTab(next);
    setParams((p) => {
      const np = new URLSearchParams(p);
      np.set("tab", next);
      return np;
    });
  };

  return (
    <div className="neo neo-auth">
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

      <div className="neo-auth-wrap">
        <div className="neo-auth-left">
          <div className="neo-auth-brand">
            ⬡ PCOS<span className="neo-auth-brand-soft">AI</span>
          </div>
          <h2 className="neo-auth-headline">
            Your health
            <br />
            data, <em>always</em>
            <br />
            <em>with you</em>
          </h2>
          <p className="neo-auth-desc">
            Save predictions, track hormonal shifts over time, and share structured reports with your doctor — all in one secure place.
          </p>
          <div className="neo-auth-features">
            <div className="neo-auth-feature">
              <div className="neo-auth-feature-dot" />
              End-to-end encrypted health data
            </div>
            <div className="neo-auth-feature">
              <div className="neo-auth-feature-dot" />
              Export reports for your doctor
            </div>
            <div className="neo-auth-feature">
              <div className="neo-auth-feature-dot" />
              Track multiple assessments over time
            </div>
            <div className="neo-auth-feature">
              <div className="neo-auth-feature-dot" />
              Completely free for individuals
            </div>
          </div>
        </div>

        <div className="neo-auth-right">
          <div className="neo-auth-card">
            <div className="neo-auth-tabs">
              <button
                type="button"
                className={`neo-auth-tab ${tab === "login" ? "active" : ""}`}
                onClick={() => setTabAndUrl("login")}
              >
                Sign In
              </button>
              <button
                type="button"
                className={`neo-auth-tab ${tab === "signup" ? "active" : ""}`}
                onClick={() => setTabAndUrl("signup")}
              >
                Create Account
              </button>
            </div>

            {tab === "login" ? (
              <div className="neo-form-panel">
                <h3 className="neo-auth-title">Welcome back</h3>
                <p className="neo-auth-sub">Sign in to access your assessments and health history.</p>

                <div className="neo-form-group">
                  <label className="neo-form-label">Email Address</label>
                  <input className="neo-form-input" type="email" placeholder="you@example.com" />
                </div>
                <div className="neo-form-group">
                  <label className="neo-form-label">Password</label>
                  <input className="neo-form-input" type="password" placeholder="••••••••••" />
                </div>
                <div className="neo-auth-forgot">
                  <a href="#" onClick={(e) => e.preventDefault()}>
                    Forgot password?
                  </a>
                </div>

                <button type="button" className="neo-auth-btn">
                  Sign In
                </button>

                <div className="neo-auth-divider">or continue with</div>
                <div className="neo-oauth-grid">
                  <button type="button" className="neo-oauth-btn">
                    Google
                  </button>
                  <button type="button" className="neo-oauth-btn">
                    GitHub
                  </button>
                </div>

                <p className="neo-auth-footer-note">
                  Don&apos;t have an account?{" "}
                  <a href="#" onClick={(e) => (e.preventDefault(), setTabAndUrl("signup"))}>
                    Create one free →
                  </a>
                </p>
              </div>
            ) : (
              <div className="neo-form-panel">
                <h3 className="neo-auth-title">Create account</h3>
                <p className="neo-auth-sub">Join thousands of women taking charge of their hormonal health.</p>

                <div className="neo-form-row">
                  <div className="neo-form-group">
                    <label className="neo-form-label">First Name</label>
                    <input className="neo-form-input" type="text" placeholder="Ayushi" />
                  </div>
                  <div className="neo-form-group">
                    <label className="neo-form-label">Last Name</label>
                    <input className="neo-form-input" type="text" placeholder="Mehra" />
                  </div>
                </div>

                <div className="neo-form-group">
                  <label className="neo-form-label">Email Address</label>
                  <input className="neo-form-input" type="email" placeholder="you@example.com" />
                </div>

                <div className="neo-form-group">
                  <label className="neo-form-label">Password</label>
                  <input
                    className="neo-form-input"
                    type="password"
                    placeholder="At least 8 characters"
                    value={signupPass}
                    onChange={(e) => setSignupPass(e.target.value)}
                  />
                </div>

                <div className="neo-strength" aria-hidden="true">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className={`neo-strength-bar ${
                        i < strength ? (strength === 1 ? "s1" : strength === 2 ? "s2" : "s3") : ""
                      }`}
                    />
                  ))}
                </div>

                <button type="button" className="neo-auth-btn">
                  Create Account
                </button>

                <div className="neo-auth-divider">or sign up with</div>
                <div className="neo-oauth-grid">
                  <button type="button" className="neo-oauth-btn">
                    Google
                  </button>
                  <button type="button" className="neo-oauth-btn">
                    GitHub
                  </button>
                </div>

                <p className="neo-auth-footer-note">
                  Already have an account?{" "}
                  <a href="#" onClick={(e) => (e.preventDefault(), setTabAndUrl("login"))}>
                    Sign in →
                  </a>
                </p>

                <p className="neo-auth-legal">
                  By creating an account you agree to our Terms of Service.
                  <br />
                  Your health data is encrypted and never sold.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

