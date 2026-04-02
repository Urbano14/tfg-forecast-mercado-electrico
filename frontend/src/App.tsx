import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import ForecastPage from "./pages/ForecastPage";
import ModelsPage from "./pages/ModelsPage";
import HistoricalPage from "./pages/historicalPage";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <nav className="app-nav">
          <div className="app-nav__brand">
            <span>TFG Energia</span>
            <span className="app-nav__brand-sub">Mercado electrico</span>
          </div>
          <div className="app-nav__links">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                isActive ? "app-nav__link app-nav__link--active" : "app-nav__link"
              }
            >
              Dashboard
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
              Forecast
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
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/historico" element={<HistoricalPage />} />
            <Route path="/forecast" element={<ForecastPage />} />
            <Route path="/models" element={<ModelsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
