import * as https from 'https';
import * as http from 'http';

export type LLMProvider = 'groq' | 'ollama';

export interface LLMConfig {
  provider: LLMProvider;
  apiKey:   string;
  model:    string;
  ollamaPort: number;
}

export async function callLLM(
  config: LLMConfig,
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  return config.provider === 'groq'
    ? callGroq(config, systemPrompt, userPrompt)
    : callOllama(config, systemPrompt, userPrompt);
}

function callGroq(
  config: LLMConfig,
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  const body = JSON.stringify({
    model: config.model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: userPrompt },
    ],
    max_tokens: 200,
    temperature: 0.35,
  });

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: 'api.groq.com',
        path:     '/openai/v1/chat/completions',
        method:   'POST',
        headers:  {
          'Content-Type':   'application/json',
          'Authorization':  `Bearer ${config.apiKey}`,
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = '';
        res.on('data', (c) => { raw += c; });
        res.on('end', () => {
          try {
            const json = JSON.parse(raw);
            if (json.error) reject(new Error(json.error.message));
            else resolve(json.choices?.[0]?.message?.content?.trim() ?? '');
          } catch {
            reject(new Error(`Groq parse error: ${raw.slice(0, 120)}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function callOllama(
  config: LLMConfig,
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  const body = JSON.stringify({
    model:    config.model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: userPrompt },
    ],
    stream: false,
  });

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: 'localhost',
        port:     config.ollamaPort,
        path:     '/api/chat',
        method:   'POST',
        headers:  {
          'Content-Type':   'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = '';
        res.on('data', (c) => { raw += c; });
        res.on('end', () => {
          try {
            const json = JSON.parse(raw);
            resolve(json.message?.content?.trim() ?? '');
          } catch {
            reject(new Error(`Ollama parse error: ${raw.slice(0, 120)}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}
