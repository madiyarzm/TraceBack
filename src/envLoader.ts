import * as fs from 'fs';
import * as path from 'path';

/**
 * Tiny zero-dependency .env loader.
 *
 * Parses `KEY=value` lines from the given file (skips blanks and `#` comments),
 * strips surrounding quotes from the value, and writes each entry into
 * `process.env` ONLY if the key isn't already set there — process-level vars
 * always take precedence over the file.
 *
 * Returns the list of keys loaded, so callers can log/diagnose.
 */
export function loadDotEnv(filePath: string): string[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, 'utf-8');
  const loaded: string[] = [];

  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const eq = line.indexOf('=');
    if (eq <= 0) continue;

    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    if (process.env[key] === undefined) {
      process.env[key] = value;
      loaded.push(key);
    }
  }

  return loaded;
}

/**
 * Tries `.env` next to the extension root and inside each open workspace
 * folder. Workspace-local .env wins because that's where users keep
 * project-specific keys.
 */
export function loadEnvFromAllKnownLocations(
  extensionRoot: string,
  workspaceFolders: readonly string[],
): string[] {
  const loaded = new Set<string>();
  for (const dir of [...workspaceFolders, extensionRoot]) {
    for (const k of loadDotEnv(path.join(dir, '.env'))) loaded.add(k);
  }
  return [...loaded];
}
