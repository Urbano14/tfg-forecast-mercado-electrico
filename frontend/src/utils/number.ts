export function formatNumber(
  value: number | null | undefined,
  decimals: number
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return value.toFixed(decimals);
}
