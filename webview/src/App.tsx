import SidebarApp from './SidebarApp';
import PanelApp from './PanelApp';

declare global {
  interface Window {
    /** Injected by the extension host before the bundle loads. */
    __TB_MODE__?: 'sidebar' | 'panel';
  }
}

/**
 * One bundle, two surfaces: the compact sidebar view and the full editor
 * panel. The extension host injects window.__TB_MODE__ into the webview HTML.
 */
export default function App() {
  return window.__TB_MODE__ === 'panel' ? <PanelApp /> : <SidebarApp />;
}
