import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import "./index.css";

import Home from "./pages/Home";
import Auth from "./pages/Auth";
import TextualPredict from "./pages/TextualPredict";
import UltrasoundPredict from "./pages/UltrasoundPredict";
import CombinedPredict from "./pages/CombinedPredict";
import Result from "./pages/Result";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/predict/textual" element={<TextualPredict />} />
        <Route path="/predict/ultrasound" element={<UltrasoundPredict />} />
        <Route path="/predict/combined" element={<CombinedPredict />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
