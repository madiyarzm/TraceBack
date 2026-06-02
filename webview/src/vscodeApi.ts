// VS Code injects a `acquireVsCodeApi()` function into the webview's global scope.
// We call it once and cache the result — calling it multiple times throws an error.
// Outside of VS Code (e.g., during local dev with `vite dev`), we return a stub.

export interface VsCodeApi {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
}

function createStubApi(): VsCodeApi {
  return {
    postMessage: (msg) => console.log('[stub vscode] postMessage:', msg),
    getState: () => ({}),
    setState: (s) => console.log('[stub vscode] setState:', s),
  };
}

const vscode: VsCodeApi =
  typeof acquireVsCodeApi !== 'undefined'
    ? acquireVsCodeApi()
    : createStubApi();

export default vscode;

// Tell TypeScript that acquireVsCodeApi exists in the webview global scope
declare function acquireVsCodeApi(): VsCodeApi;
