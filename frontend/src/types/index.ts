export interface ExplainResult {
  overview: string;
  functions: { name: string; purpose: string; parameters: string; returns: string; logic: string; docstring: string }[];
  flowchart: string;
  key_concepts: string[];
  potential_issues: string[];
  complexity: string;
  error?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export type Tab = 'docs' | 'debug' | 'chat';
export type DebugMode = 'analyze' | 'multimodal' | 'decode' | 'custom';

export interface IndexedFile {
  filename: string;
  filepath: string;
  language: string;
}
