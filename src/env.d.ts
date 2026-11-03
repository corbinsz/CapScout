interface ImportMetaEnv {
  readonly VITE_API_URL?: string
  readonly VITE_ANALYTICS?: string
  readonly VITE_ERROR_REPORTING?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
