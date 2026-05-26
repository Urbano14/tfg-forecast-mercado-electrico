// Formatea un número con un número fijo de decimales, muestra precios, métricas y valores numéricos de forma uniforme en la interfaz.
export function formatNumber(
  value: number | null | undefined,
  decimals: number
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  // Ejemplo: formatNumber(12.3456, 2) -> "12.35".
  return value.toFixed(decimals);
}