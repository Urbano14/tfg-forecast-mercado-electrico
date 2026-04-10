## Puesta en marcha (resumen)

1. Iniciar la aplicación:
   ```powershell
   docker compose up -d --build
   ```

2. Primera vez (carga de datos):
   ```powershell
   docker compose exec backend python -m scripts.load_market_data
   ```

3. Accesos:
   - Frontend: http://localhost:3001
   - Backend: http://localhost:8000/api/v1

4. Parar todo:
   ```powershell
   docker compose down
   ```
