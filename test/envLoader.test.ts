import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { loadDotEnv } from '../src/envLoader';

let tmpDir: string;
let envPath: string;

const SAVED = {
  GROQ_API_KEY: process.env.GROQ_API_KEY,
  GROQ_MODEL:   process.env.GROQ_MODEL,
  TB_FOO:       process.env.TB_FOO,
};

beforeEach(() => {
  tmpDir  = fs.mkdtempSync(path.join(os.tmpdir(), 'tb-env-'));
  envPath = path.join(tmpDir, '.env');
  delete process.env.GROQ_API_KEY;
  delete process.env.GROQ_MODEL;
  delete process.env.TB_FOO;
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  for (const [k, v] of Object.entries(SAVED)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
});

describe('loadDotEnv', () => {
  it('returns [] when the file does not exist', () => {
    expect(loadDotEnv(path.join(tmpDir, 'missing.env'))).toEqual([]);
  });

  it('loads KEY=value pairs into process.env', () => {
    fs.writeFileSync(envPath, 'GROQ_API_KEY=gsk_abc\nTB_FOO=bar\n');
    const loaded = loadDotEnv(envPath);
    expect(loaded).toContain('GROQ_API_KEY');
    expect(loaded).toContain('TB_FOO');
    expect(process.env.GROQ_API_KEY).toBe('gsk_abc');
    expect(process.env.TB_FOO).toBe('bar');
  });

  it('skips blank lines and # comments', () => {
    fs.writeFileSync(envPath, '# header\n\nGROQ_MODEL=llama-3.1\n');
    loadDotEnv(envPath);
    expect(process.env.GROQ_MODEL).toBe('llama-3.1');
  });

  it('strips matching surrounding quotes from values', () => {
    fs.writeFileSync(envPath, 'TB_FOO="quoted value"\n');
    loadDotEnv(envPath);
    expect(process.env.TB_FOO).toBe('quoted value');
  });

  it('does NOT overwrite a process.env value already set', () => {
    process.env.GROQ_API_KEY = 'from-shell';
    fs.writeFileSync(envPath, 'GROQ_API_KEY=from-file\n');
    const loaded = loadDotEnv(envPath);
    expect(process.env.GROQ_API_KEY).toBe('from-shell');
    expect(loaded).not.toContain('GROQ_API_KEY');
  });
});
