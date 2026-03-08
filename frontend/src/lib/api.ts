import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000, // 30s for normal requests
});

// Ingestion gets a much longer timeout — cold start + downloading files
export const ingestApi = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 minutes for ingestion
});

// Session management
export const createSession = () => api.post('/session/create');
export const getSessionStats = (sessionId: string) => api.get(`/session/${sessionId}/stats`);
export const clearSession = (sessionId: string) => api.delete(`/session/${sessionId}`);

// Ingestion — uses long timeout client
export const ingestGithub = (repoUrl: string, sessionId: string, githubToken?: string) =>
  ingestApi.post('/ingest/github', { repo_url: repoUrl, session_id: sessionId, github_token: githubToken });

export const ingestFiles = (files: { filename: string; content: string }[], sessionId: string) =>
  ingestApi.post('/ingest/files', { files, session_id: sessionId });

// Documentation
export const explainCode = (code: string, language: string) =>
  api.post('/docs/explain', { code, language });

export const explainCodeCommented = (code: string, language: string) =>
  api.post('/docs/commented-code', { code, language });

export const fetchFileContent = (sessionId: string, filepath: string) =>
  api.get('/fetch-file', { params: { session_id: sessionId, filepath } });

// Debugging
export const debugAnalyze = (data: {
  stacktrace?: string;
  error_message?: string;
  code_context?: string;
  console_logs?: string;
}) => api.post('/debug/analyze', data);

export const multimodalDebug = (data: {
  stacktrace?: string;
  console_logs?: string;
  screenshot_base64?: string;
}) => api.post('/debug/multimodal', data);

export const decodeStacktrace = (stacktrace: string, sourceMap?: string) =>
  api.post('/debug/decode-stacktrace', { stacktrace, source_map: sourceMap });

// Chat
export const sendChat = (query: string, sessionId: string, history?: any[]) =>
  api.post('/chat', { query, session_id: sessionId, conversation_history: history });

// Health
export const checkHealth = () => api.get('/health');
