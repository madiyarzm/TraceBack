// In-memory code → URL store. Swap for Redis/Postgres behind the same surface.

const byCode = new Map();

export function save(code, url) {
  byCode.set(code, url);
  return code;
}

export function resolve(code) {
  return byCode.get(code) ?? null;
}

export function has(code) {
  return byCode.has(code);
}

export function size() {
  return byCode.size;
}

export function reset() {
  byCode.clear();
}
