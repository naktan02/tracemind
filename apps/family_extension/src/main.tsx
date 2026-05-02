import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./styles.css";

const rootElement = document.getElementById("root");

if (!(rootElement instanceof HTMLElement)) {
  throw new Error("family_extension root element를 찾지 못했습니다.");
}

const initialRoute = rootElement.dataset.initialRoute;

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App initialRoute={initialRoute} />
  </React.StrictMode>,
);
