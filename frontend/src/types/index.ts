export interface FunctionDoc {
  name: string;
  purpose: string;
  parameters: string;
  returns: string;
  logic: string;
  docstring: string;
}

export interface ExplainResult {
  overview: string;
  functions: FunctionDoc[];
  flowchart: string;
  key_concepts: string[];
  potential_issues: string[];
  complexity: string;
  error?: string;
}

export interface DebugResult {
  error_type: string;
  root_cause: string;
  explanation: string;
  fix: string;
  fixed_code: string;
  prevention: string;
  related_issues: string[];
  confidence: 'high' | 'medium' | 'low';
  error?: string;
}

export interface MultimodalResult {
  most_likely_file: string;
  most_likely_line: string;
  root_cause: string;
  signal_correlation: string;
  fix_steps: string[];
  backend_vs_frontend: string;
  screenshot_interpretation?: string;
  error?: string;
}

export interface DecodeResult {
  decoded_frames: { original: string; decoded: string; confidence: string }[];
  likely_source_files: string[];
  entry_point: string;
  summary: string;
  error?: string;
}

export interface IndexedFile {
  filename: string;
  filepath: string;
  language: string;
}

export interface SessionStats {
  total_files: number;
  files: IndexedFile[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export type Tab = 'docs' | 'debug' | 'chat';
export type DocMode = 'explain' | 'flowchart';
export type DebugMode = 'analyze' | 'multimodal' | 'decode';
