import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import ForecastPage from "./pages/ForecastPage";
import ModelsPage from "./pages/ModelsPage";
import HistoricalPage from "./pages/historicalPage";
import "./App.css";

// Componente raíz del frontend. Solo se define la estructura general,
// la navegación superior y qué página se muestra según la ruta actual.
function App() {
  return (
    // BrowserRouter permite cambiar entre /, /historico, /forecast y /models sin recargar toda la página.
    <BrowserRouter>
      <div className="app-shell">
        <div className="app-container">
          <nav className="app-nav">
            <div className="app-nav__brand">
              <span className="app-nav__brand-mark" aria-hidden="true">
                {"\u26A1"}
              </span>
              <div className="app-nav__brand-copy">
                <span className="app-nav__brand-title">TFG Energia</span>
                <span className="app-nav__brand-sub">Mercado electrico</span>
              </div>
            </div>

            {/* Enlaces de navegación entre las páginas principales. */}
            <div className="app-nav__links">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  isActive ? "app-nav__link app-nav__link--active" : "app-nav__link"
                }
              >
                Inicio
              </NavLink>

              <NavLink
                to="/historico"
                className={({ isActive }) =>
                  isActive ? "app-nav__link app-nav__link--active" : "app-nav__link"
                }
              >
                Historico
              </NavLink>

              <NavLink
                to="/forecast"
                className={({ isActive }) =>
                  isActive ? "app-nav__link app-nav__link--active" : "app-nav__link"
                }
              >
                Prediccion
              </NavLink>

              <NavLink
                to="/models"
                className={({ isActive }) =>
                  isActive ? "app-nav__link app-nav__link--active" : "app-nav__link"
                }
              >
                Modelos
              </NavLink>
            </div>
          </nav>

          {/* Zona principal donde se renderiza la página que corresponda a la ruta actual. */}
          <main className="app-main">
            <Routes>
              <Route path="/" element={<DashboardPage />} />

              <Route path="/historico" element={<HistoricalPage />} />

              <Route path="/forecast" element={<ForecastPage />} />

              <Route path="/models" element={<ModelsPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
