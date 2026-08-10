import { save, resolve, has } from './store.js';

const ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

// BUG: a random 4-char code has a birthday-paradox collision problem — with a
// few thousand links, two URLs can be handed the same code and the first one
// gets silently overwritten. The demo's task is to make codes collision-free.
function randomCode() {
  let code = '';
  for (let i = 0; i < 4; i++) {
    code += ALPHABET[Math.floor(Math.random() * ALPHABET.length)];
  }
  return code;
}

export function shorten(url) {
  if (typeof url !== 'string' || !url.startsWith('http')) {
    throw new Error('shorten() requires an absolute http(s) URL');
  }
  const code = randomCode();
  return save(code, url);
}

export function expand(code) {
  return resolve(code);
}

export { has };
