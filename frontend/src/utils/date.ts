export const APP_TIMEZONE = "Europe/Madrid";

const dateFormatter = new Intl.DateTimeFormat("es-ES", {
  timeZone: APP_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

const dateTimeFormatter = new Intl.DateTimeFormat("es-ES", {
  timeZone: APP_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function getPart(parts: Intl.DateTimeFormatPart[], type: string): string {
  return parts.find((part) => part.type === type)?.value ?? "";
}

function parseDateValue(value: string | Date): Date | null {
  const date = value instanceof Date ? value : new Date(value);

  if (Number.isNaN(date.getTime())) {
    return null;
  }

  return date;
}

export function formatDate(value: string | Date): string {
  const date = parseDateValue(value);
  if (!date) {
    return "-";
  }

  const parts = dateFormatter.formatToParts(date);
  const day = getPart(parts, "day");
  const month = getPart(parts, "month");
  const year = getPart(parts, "year");

  return `${day}/${month}/${year}`;
}

export function formatTimestamp(value: string | Date): string {
  const date = parseDateValue(value);
  if (!date) {
    return "-";
  }

  const parts = dateTimeFormatter.formatToParts(date);
  const day = getPart(parts, "day");
  const month = getPart(parts, "month");
  const year = getPart(parts, "year");
  const hours = getPart(parts, "hour");
  const minutes = getPart(parts, "minute");

  return `${day}/${month}/${year} ${hours}:${minutes}`;
}

export function toDateInputValue(value: string): string {
  if (!value) {
    return "";
  }

  if (value.includes("T")) {
    return value.split("T")[0];
  }

  if (value.length >= 10 && /^\d{4}-\d{2}-\d{2}/.test(value)) {
    return value.slice(0, 10);
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "";
  }

  return parsed.toISOString().slice(0, 10);
}

export function formatLocalDateTimeInput(value: string): string {
  if (!value) {
    return "";
  }

  const [date, time] = value.split("T");
  if (!time) {
    return formatDate(date);
  }

  return `${formatDate(date)} ${time.slice(0, 5)}`;
}
