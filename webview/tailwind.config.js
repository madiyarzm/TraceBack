/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // VS Code's CSS variables exposed as Tailwind tokens
        'vsc-bg':          'var(--vscode-editor-background)',
        'vsc-fg':          'var(--vscode-editor-foreground)',
        'vsc-border':      'var(--vscode-panel-border)',
        'vsc-accent':      'var(--vscode-focusBorder)',
        'vsc-sidebar-bg':  'var(--vscode-sideBar-background)',
      },
    },
  },
  plugins: [],
};
