import * as fs from 'fs';

/**
 * Reads REAL token usage from a Claude Code transcript (JSONL).
 *
 * Every hook payload carries `transcript_path`; each assistant message line
 * in that file embeds `message.usage`. The latest one gives the session's
 * current context size: input + cache_read + cache_creation + output.
 *
 * Only the tail of the file is read (transcripts grow to MBs), so a refresh
 * is O(1) in transcript size.
 */

const TAIL_BYTES = 256 * 1024;

export async function readContextTokens(transcriptPath: string): Promise<number | null> {
  try {
    const stat  = await fs.promises.stat(transcriptPath);
    const start = Math.max(0, stat.size - TAIL_BYTES);
    const fh    = await fs.promises.open(transcriptPath, 'r');
    const buf   = Buffer.alloc(stat.size - start);
    await fh.read(buf, 0, buf.length, start);
    await fh.close();

    const lines = buf.toString('utf-8').split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      if (!lines[i].includes('"usage"')) continue;
      try {
        const usage = JSON.parse(lines[i])?.message?.usage;
        if (usage && typeof usage.input_tokens === 'number') {
          return (
            (usage.input_tokens ?? 0) +
            (usage.cache_read_input_tokens ?? 0) +
            (usage.cache_creation_input_tokens ?? 0) +
            (usage.output_tokens ?? 0)
          );
        }
      } catch {
        // partial line at the chunk boundary or non-JSON content — keep scanning
      }
    }
    return null;
  } catch {
    return null; // file missing/unreadable — caller falls back to estimates
  }
}
