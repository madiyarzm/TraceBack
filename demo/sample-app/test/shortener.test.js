import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shorten, expand } from '../src/shortener.js';
import { reset } from '../src/store.js';

test('shorten returns a code that expands back to the URL', () => {
  reset();
  const code = shorten('https://example.com/a-long-path');
  assert.equal(expand(code), 'https://example.com/a-long-path');
});

test('shorten rejects non-http input', () => {
  reset();
  assert.throws(() => shorten('not-a-url'));
});
