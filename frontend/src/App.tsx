import { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Github, Upload, MessageSquare, Bug, FileText, ChevronRight,
  Zap, Copy, Check, AlertTriangle, CheckCircle2,
  XCircle, Loader2, Image, Code2, GitBranch, Layers, Search,
  Terminal, Sparkles, ChevronDown, ChevronUp, FolderOpen, FileCode2,
} from 'lucide-react';
import mermaid from 'mermaid';
import {
  createSession, ingestGithub, ingestFiles, explainCode, explainCodeCommented, fetchFileContent,
  debugAnalyze, multimodalDebug, decodeStacktrace, sendChat, checkHealth,
} from './lib/api';
import type { ExplainResult, ChatMessage, Tab, DebugMode, IndexedFile } from './types';
import './index.css';

mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });

const LANGS = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'csharp', 'ruby', 'php'];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="copy-btn">
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

function Spinner() { return <Loader2 size={16} className="spin" />; }

function Badge({ text, type = 'default' }: { text: string; type?: string }) {
  return <span className={`badge badge-${type}`}>{text}</span>;
}

function MermaidDiagram({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || !chart) return;
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    mermaid.render(id, chart)
      .then(({ svg }) => {
        if (ref.current) ref.current.innerHTML = svg;
      })
      .catch(() => {
        // On error: show raw source as readable code, no bomb icon
        if (ref.current) {
          ref.current.innerHTML = `
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px">
              <div style="color:#8b9ab8;font-size:11px;margin-bottom:6px">⚠ Diagram preview unavailable — raw source:</div>
              <pre style="color:#79c0ff;font-size:12px;margin:0;overflow:auto;white-space:pre-wrap">${chart.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre>
            </div>`;
        }
      });
  }, [chart]);
  return <div ref={ref} className="mermaid-wrap" />;
}

function Collapsible({ title, children, defaultOpen = false, icon }: any) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible">
      <button className="coll-hdr" onClick={() => setOpen((o: boolean) => !o)}>
        <span className="coll-title">{icon && <span style={{ marginRight: 6 }}>{icon}</span>}{title}</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && <div className="coll-body">{children}</div>}
    </div>
  );
}

// ── Ingestion Panel ─────────────────────────────────────────────────────────────
function sanitizeMermaid(chart: string): string {
  if (!chart) return '';
  let c = chart.trim();

  // Strip any markdown fences (```mermaid, ```graph, etc.)
  c = c.replace(/^```[\w]*\s*/i, '').replace(/```\s*$/, '').trim();

  // Detect diagram type
  const isSequence = /^sequenceDiagram/i.test(c);
  const isFlowchart = /^(flowchart|graph)/i.test(c);

  // Add default type if missing
  if (!c.match(/^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph)/i)) {
    c = 'flowchart TD\n' + c;
  }

  if (isSequence) {
    // sequenceDiagram: only fix single-line collapse, leave arrows alone (>> is valid syntax)
    const lineCount = c.split('\n').filter((l: string) => l.trim()).length;
    if (lineCount <= 2) {
      // Split on sequence arrow types: ->>, -->>, ->, -->, -x, --x
      c = c.replace(/\s*(->>|-->>|->|-->|-x|--x)\s*/g, ' $1\n    ');
      c = c.replace(/(sequenceDiagram)\s+/, '$1\n');
    }
    // Fix quoted labels: remove outer quotes that break sequence parser
    c = c.replace(/as "(.*?)"/g, 'as $1');
    return c;
  }

  if (isFlowchart) {
    // Flowchart: fix single-line, clean labels
    const lineCount = c.split('\n').filter((l: string) => l.trim()).length;
    if (lineCount <= 2) {
      c = c.replace(/(flowchart\s+\w+|graph\s+\w+)/i, '$1\n');
      c = c.replace(/\s*(-->|---|-.->|==>|-\.-?>)\s*/g, ' $1\n  ');
      c = c.replace(/\n{2,}/g, '\n');
    }
    // Remove parentheses inside [] labels — Mermaid treats () as subgraph call
    c = c.replace(/\[([^\]]*)\(([^\)]*)\)([^\]]*)\]/g, '[$1 $2 $3]');
    // Strip chars that break flowchart node label parsing (but NOT sequence arrows)
    c = c.replace(/\[([^\]]+)\]/g, (_m: string, inner: string) => {
      const clean = inner.replace(/[<>{}"|]/g, '').replace(/\s+/g, ' ').trim();
      return '[' + clean + ']';
    });
  }

  // Fix typographic em/en dashes used as arrow dashes
  c = c.replace(/\u2014>/g, '-->').replace(/\u2013>/g, '-->');

  return c;
}

// ── GitHub-style folder tree ───────────────────────────────────────────────────
function FolderTree({ files, selectedFile, onSelect }: {
  files: IndexedFile[];
  selectedFile: IndexedFile | null;
  onSelect: (f: IndexedFile) => void;
}) {
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set(['root']));
  const seen = new Set<string>();
  const uniqueFiles = files
    .map(f => ({ ...f, filename: f.filename.replace(/\s+chunk\s+\d+\/\d+$/i, '').trim() }))
    .filter(f => {
      if (seen.has(f.filepath)) return false;
      seen.add(f.filepath);
      return true;
    });
  const tree: Record<string, IndexedFile[]> = {};
  uniqueFiles.forEach(f => {
    const parts = f.filepath.split('/');
    const folder = parts.length > 1 ? parts.slice(0, -1).join('/') : 'root';
    if (!tree[folder]) tree[folder] = [];
    tree[folder].push(f);
  });
  const toggleFolder = (folder: string) => {
    setOpenFolders(prev => {
      const next = new Set(prev);
      if (next.has(folder)) next.delete(folder); else next.add(folder);
      return next;
    });
  };
  const extIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const map: Record<string, string> = { py: '🐍', js: '🟨', ts: '🔷', jsx: '⚛️', tsx: '⚛️', java: '☕', go: '🐹', rs: '🦀', css: '🎨', html: '🌐', json: '📋', md: '📝' };
    return map[ext] || '📄';
  };
  return (
    <div className="folder-tree">
      {Object.entries(tree).sort(([a], [b]) => a === 'root' ? -1 : a.localeCompare(b)).map(([folder, folderFiles]) => (
        <div key={folder} className="tree-folder">
          {folder !== 'root' && (
            <button className="tree-folder-btn" onClick={() => toggleFolder(folder)}>
              <span className="tree-arrow">{openFolders.has(folder) ? '▾' : '▸'}</span>
              <span className="tree-folder-ico">📁</span>
              <span className="tree-folder-name">{folder.split('/').pop()}</span>
              <span className="tree-count">{folderFiles.length}</span>
            </button>
          )}
          {(folder === 'root' || openFolders.has(folder)) && (
            <div className={`tree-files ${folder !== 'root' ? 'indented' : ''}`}>
              {folderFiles.map((f, i) => (
                <button
                  key={i}
                  className={`tree-file-btn ${selectedFile?.filepath === f.filepath ? 'active' : ''}`}
                  onClick={() => onSelect(f)}
                >
                  <span className="tree-file-ico">{extIcon(f.filename)}</span>
                  <span className="tree-file-name">{f.filename}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Progress steps shown during indexing
const INGEST_STEPS = [
  'Fetching file tree from GitHub...',
  'Filtering and downloading files...',
  'Chunking code into segments...',
  'Generating embeddings...',
  'Building vector index...',
  'Finalizing session...',
];

function IngestProgress({ startTime }: { startTime: number }) {
  const [step, setStep] = useState(0);
  const [pct, setPct] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    // Simulate realistic progress: fast at start, slows near end
    const schedule = [
      { at: 800,  pct: 12, step: 0 },
      { at: 2000, pct: 28, step: 1 },
      { at: 4000, pct: 45, step: 2 },
      { at: 7000, pct: 62, step: 3 },
      { at: 11000,pct: 78, step: 4 },
      { at: 16000,pct: 90, step: 5 },
    ];
    const timers = schedule.map(({ at, pct: p, step: s }) =>
      setTimeout(() => { setPct(p); setStep(s); }, at)
    );
    const ticker = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 500);
    return () => { timers.forEach(clearTimeout); clearInterval(ticker); };
  }, [startTime]);

  const estTotal = 25; // seconds estimate
  const remaining = Math.max(0, estTotal - elapsed);

  return (
    <div className="ingest-progress">
      <div className="ip-header">
        <span className="ip-step-label"><Spinner /> {INGEST_STEPS[step]}</span>
        <span className="ip-pct">{pct}%</span>
      </div>
      <div className="ip-bar-track">
        <div className="ip-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="ip-footer">
        <span className="ip-elapsed">⏱ {elapsed}s elapsed</span>
        <span className="ip-remain">{remaining > 0 ? `~${remaining}s remaining` : 'Almost done...'}</span>
      </div>
      <div className="ip-steps">
        {INGEST_STEPS.map((s, i) => (
          <div key={i} className={`ip-step-dot ${i < step ? 'done' : i === step ? 'active' : ''}`}>
            <span className="ip-dot">{i < step ? '✓' : i === step ? '●' : '○'}</span>
            <span className="ip-step-name">{s.replace('...', '')}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function IngestionPanel({
  sessionId, onIndexed, apiOk,
}: {
  sessionId: string;
  apiOk: boolean | null;
  onIndexed: (n: number, files: IndexedFile[], repoInfo?: {owner:string,repo:string,branch:string}) => void;
}) {
  const [tab, setTab] = useState<'gh' | 'up'>('gh');
  const [url, setUrl] = useState('');
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [startTime, setStartTime] = useState(0);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState('');

  const doGithub = async () => {
    if (!url.trim() || apiOk !== true) return;
    setLoading(true); setErr(''); setRes(null);
    setStartTime(Date.now());
    try {
      const r = await ingestGithub(url.trim(), sessionId, token || undefined);
      setRes(r.data);
      onIndexed(r.data.indexed_files, r.data.files || [], {
        owner: url.trim().replace(/\/+$/, '').split('/').slice(-2)[0],
        repo:  url.trim().replace(/\/+$/, '').split('/').slice(-1)[0],
        branch: r.data.branch || 'main',
      });
    } catch (e: any) { setErr(e.response?.data?.detail || e.message); }
    finally { setLoading(false); }
  };

  const doUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setLoading(true); setErr(''); setRes(null);
    setStartTime(Date.now());
    try {
      const data = await Promise.all(files.map(async f => ({ filename: f.name, content: await f.text() })));
      const r = await ingestFiles(data, sessionId);
      setRes(r.data);
      onIndexed(r.data.indexed_files, r.data.files || []);
    } catch (e: any) { setErr(e.response?.data?.detail || e.message); }
    finally { setLoading(false); }
  };

  const isApiReady = apiOk === true;

  return (
    <div className="ingest">
      <div className="ingest-tabs">
        <button className={`itab ${tab === 'gh' ? 'active' : ''}`} onClick={() => setTab('gh')}><Github size={13} /> GitHub</button>
        <button className={`itab ${tab === 'up' ? 'active' : ''}`} onClick={() => setTab('up')}><Upload size={13} /> Upload</button>
      </div>
      {tab === 'gh' && (
        <div className="ingest-row">
          <input className="inp" placeholder="https://github.com/owner/repo" value={url} onChange={e => setUrl(e.target.value)} onKeyDown={e => e.key === 'Enter' && isApiReady && !loading && doGithub()} disabled={loading} />
          <input className="inp" type="password" placeholder="GitHub Token (optional)" value={token} onChange={e => setToken(e.target.value)} disabled={loading} />
          <button
            className={`btn-pri ${!isApiReady ? 'btn-disabled' : ''}`}
            onClick={doGithub}
            disabled={loading || !url || !isApiReady}
            title={!isApiReady ? (apiOk === false ? 'API is offline' : 'Waiting for API to connect...') : ''}
          >
            {loading ? <Spinner /> : <GitBranch size={14} />}
            {loading ? 'Indexing...' : !isApiReady ? (apiOk === null ? 'Connecting...' : 'API Offline') : 'Index Repo'}
          </button>
        </div>
      )}
      {tab === 'up' && (
        <label className={`upload-zone ${!isApiReady ? 'upload-disabled' : ''}`}>
          <input type="file" multiple accept=".py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.cpp,.c,.md,.txt,.json" onChange={isApiReady ? doUpload : undefined} hidden disabled={!isApiReady} />
          {!isApiReady
            ? <><Spinner /> <span>{apiOk === null ? 'Waiting for API...' : 'API Offline'}</span></>
            : <><Upload size={22} /> <span>Click to upload code files</span></>
          }
        </label>
      )}
      {loading && <IngestProgress startTime={startTime} />}
      {err && <div className="err-row"><XCircle size={13} /> {err}</div>}
      {res && !loading && <div className="ok-row"><CheckCircle2 size={13} /> Indexed <b>{res.indexed_files}</b> files{res.repo ? ` from ${res.repo}` : ''}</div>}
    </div>
  );
}

// ── Docs Page — documents the INDEXED REPO ─────────────────────────────────────
function DocsPage({ sessionId, indexedFiles, indexedCount, repoInfo: repoProp }: {
  sessionId: string;
  indexedFiles: IndexedFile[];
  indexedCount: number;
  repoInfo: {owner:string,repo:string,branch:string} | null;
}) {
  const [selectedFile, setSelectedFile] = useState<IndexedFile | null>(null);
  const [customCode, setCustomCode] = useState('');
  const [lang, setLang] = useState('python');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [err, setErr] = useState('');
  const [view, setView] = useState<'overview' | 'functions' | 'flowchart' | 'code'>('overview');
  const [commentedCode, setCommentedCode] = useState<string>('');
  const [commentedLoading, setCommentedLoading] = useState(false);
  const [lastCodeForComment, setLastCodeForComment] = useState('');
  const [mode, setMode] = useState<'repo' | 'custom'>('repo');
  useEffect(() => { if (repoProp) setRepoInfo(repoProp); }, [repoProp]);
  const [repoInfo, setRepoInfo] = useState<{owner:string,repo:string,branch:string} | null>(repoProp);
  const [currentCode, setCurrentCode] = useState('');
  const [currentLang, setCurrentLang] = useState('python');

  const analyzeFile = async (file: IndexedFile) => {
    setSelectedFile(file);
    setLoading(true); setErr(''); setResult(null);
    try {
      let code = '';
      // Try to fetch real file from GitHub raw URL directly
      if (repoInfo) {
        const rawUrl = `https://raw.githubusercontent.com/${repoInfo.owner}/${repoInfo.repo}/${repoInfo.branch}/${file.filepath}`;
        try {
          const res = await fetch(rawUrl);
          if (res.ok) code = await res.text();
        } catch { /* ignore, fall through */ }
      }
      // Fallback: try backend fetch-file endpoint
      if (!code) {
        try {
          const fileRes = await fetchFileContent(sessionId, file.filepath);
          code = fileRes.data.content;
        } catch { /* ignore */ }
      }
      if (!code) {
        setErr('Could not fetch file content. Make sure the repo is public and re-index it.');
        setLoading(false);
        return;
      }
      const r = await explainCode(code, file.language);
      if (r.data.error) setErr(r.data.error);
      else { setResult(r.data); setView('overview'); setCurrentCode(code); setCurrentLang(file.language); setCommentedCode(''); }
    } catch (e: any) { setErr(e.response?.data?.detail || e.message || 'Error'); }
    finally { setLoading(false); }
  };

  const loadCommentedCode = async (code: string, language: string) => {
    if (commentedCode && lastCodeForComment === code) return; // already loaded
    setCommentedLoading(true);
    try {
      const r = await explainCodeCommented(code, language);
      setCommentedCode(r.data.commented_code || '');
      setLastCodeForComment(code);
    } catch { setCommentedCode('// Error loading commented code'); }
    finally { setCommentedLoading(false); }
  };

  const analyzeCustom = async () => {
    if (!customCode.trim()) return;
    setLoading(true); setErr(''); setResult(null); setSelectedFile(null);
    try {
      const r = await explainCode(customCode, lang);
      if (r.data.error) setErr(r.data.error);
      else { setResult(r.data); setView('overview'); setCurrentCode(customCode); setCurrentLang(lang); setCommentedCode(''); }
    } catch (e: any) { setErr(e.response?.data?.detail || e.message || 'Error'); }
    finally { setLoading(false); }
  };

  return (
    <div className="split">
      {/* Left — folder tree or custom editor */}
      <div className="split-left">
        <div className="doc-mode-tabs">
          <button className={`dtab ${mode === 'repo' ? 'active' : ''}`} onClick={() => setMode('repo')}><FolderOpen size={13} /> Repo Files</button>
          <button className={`dtab ${mode === 'custom' ? 'active' : ''}`} onClick={() => setMode('custom')}><Code2 size={13} /> Custom Code</button>
        </div>

        {mode === 'repo' && (
          <div className="file-list-wrap">
            {indexedCount === 0 ? (
              <div className="no-files">
                <FolderOpen size={32} className="empty-ico" />
                <p>No files indexed yet.</p>
                <p className="muted-c">Use the GitHub or Upload button above to index a repo first.</p>
              </div>
            ) : (
              <>
                <div className="file-list-hdr"><FileCode2 size={13} /> {indexedCount} files indexed — click any file to document it</div>
                <FolderTree files={indexedFiles} selectedFile={selectedFile} onSelect={analyzeFile} />
              </>
            )}
          </div>
        )}

        {mode === 'custom' && (
          <div className="custom-doc-wrap">
            <div className="split-hdr">
              <div className="split-title"><Code2 size={15} /> Paste Code</div>
              <select className="sel" value={lang} onChange={e => setLang(e.target.value)}>
                {LANGS.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>
            <div className="editor-box">
              <Editor height="100%" language={lang} value={customCode} onChange={v => setCustomCode(v || '')}
                theme="vs-dark" options={{ fontSize: 13, minimap: { enabled: false }, wordWrap: 'on', scrollBeyondLastLine: false }} />
            </div>
            <button className="btn-pri full" onClick={analyzeCustom} disabled={loading || !customCode.trim()}>
              {loading ? <><Spinner /> Analyzing...</> : <><Sparkles size={14} /> Analyze & Document</>}
            </button>
          </div>
        )}
      </div>

      {/* Right — rich results */}
      <div className="split-right">
        {!result && !loading && !err && (
          <div className="empty">
            <Sparkles size={42} className="empty-ico" />
            <h3>Documentation Helper</h3>
            {mode === 'repo'
              ? <p>Index a repo above, then click any file on the left to generate full docs for it.</p>
              : <p>Paste any code on the left and click Analyze to generate docs, docstrings, and a flowchart.</p>
            }
            <div className="feat-list">
              {['README-style overview', 'Architecture explanation', 'API documentation', 'Function-by-function breakdown', 'Commented source code', 'Mermaid flowchart'].map(f => (
                <div key={f} className="feat"><CheckCircle2 size={13} /> {f}</div>
              ))}
            </div>
          </div>
        )}
        {loading && (
          <div className="loading">
            <Spinner />
            <p>Generating full documentation for {selectedFile ? selectedFile.filename : 'code'}...</p>
          </div>
        )}
        {err && <div className="err-box"><XCircle size={14} /> {err}</div>}
        {result && (
          <div className="result">
            {selectedFile && (
              <div className="result-file-hdr">
                <FileCode2 size={14} /> <b>{selectedFile.filename}</b>
                <span className="muted-c">{selectedFile.filepath}</span>
              </div>
            )}
            <div className="rtabs">
              <button className={`rtab ${view === 'overview' ? 'active' : ''}`} onClick={() => setView('overview')}><FileText size={12} /> Overview</button>
              <button className={`rtab ${view === 'functions' ? 'active' : ''}`} onClick={() => setView('functions')}><Code2 size={12} /> Functions ({result.functions?.length || 0})</button>
              <button className={`rtab ${view === 'code' ? 'active' : ''}`} onClick={() => { setView('code'); loadCommentedCode(currentCode, currentLang); }}><Terminal size={12} /> Commented Code</button>
              <button className={`rtab ${view === 'flowchart' ? 'active' : ''}`} onClick={() => setView('flowchart')}><GitBranch size={12} /> Flowchart</button>
            </div>
            <div className="rcontent">

              {view === 'overview' && (
                <>
                  <div className="card blue-card">
                    <h4>📋 What it does</h4>
                    <p>{result.overview}</p>
                  </div>
                  {(result as any).readme && (
                    <div className="card green-card">
                      <h4>📖 README</h4>
                      <p>{(result as any).readme}</p>
                    </div>
                  )}
                  {(result as any).architecture && (
                    <div className="card purple-card">
                      <h4>🏗️ Architecture</h4>
                      <p>{(result as any).architecture}</p>
                    </div>
                  )}
                  {(result as any).api_docs && (
                    <div className="card">
                      <h4>🔌 API Documentation</h4>
                      <pre className="api-docs-block">{(result as any).api_docs}</pre>
                    </div>
                  )}
                  {result.key_concepts?.length > 0 && (
                    <div>
                      <div style={{fontSize:12,color:'var(--muted)',marginBottom:6}}>KEY CONCEPTS</div>
                      <div className="tags">{result.key_concepts.map(c => <Badge key={c} text={c} />)}</div>
                    </div>
                  )}
                  {result.complexity && <div className="info-card"><b>⚡ Complexity:</b> {result.complexity}</div>}
                  {result.potential_issues?.length > 0 && (
                    <div className="warn-card">
                      <b><AlertTriangle size={13} /> Potential Issues</b>
                      <ul>{result.potential_issues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                    </div>
                  )}
                </>
              )}

              {view === 'functions' && (
                <>
                  {(!result.functions || result.functions.length === 0) && (
                    <div className="muted-c">No functions found in this file.</div>
                  )}
                  <div className="fn-list-hdr">
                    <span>{result.functions.length} functions found — click to expand</span>
                  </div>
                  {result.functions?.map((fn, i) => (
                    <Collapsible key={i} title={fn.name} defaultOpen={false} icon={<Code2 size={13} />}>
                      <div className="fn-rows">
                        {fn.purpose && <div><b>Purpose:</b> {fn.purpose}</div>}
                        {fn.parameters && <div><b>Parameters:</b> {fn.parameters}</div>}
                        {fn.returns && <div><b>Returns:</b> {fn.returns}</div>}
                        {(fn as any).complexity && <div><b>Complexity:</b> {(fn as any).complexity}</div>}
                        {fn.logic && <div><b>Logic:</b> {fn.logic}</div>}
                        {fn.docstring && (
                          <div className="docblock">
                            <div className="docblock-hdr"><b>Docstring</b><CopyButton text={fn.docstring} /></div>
                            <pre className="codeblock">{fn.docstring}</pre>
                          </div>
                        )}
                      </div>
                    </Collapsible>
                  ))}
                </>
              )}

              {view === 'code' && (
                <>
                  {commentedLoading && <div className="loading"><Spinner /><p>Generating commented code...</p></div>}
                  {!commentedLoading && commentedCode && (
                    <div className="commented-code-wrap">
                      <div className="flow-hdr">
                        <span>📝 Commented Source Code</span>
                        <CopyButton text={commentedCode} />
                      </div>
                      <pre className="commented-code">{commentedCode}</pre>
                    </div>
                  )}
                  {!commentedLoading && !commentedCode && (
                    <div className="muted-c" style={{padding:20}}>Click the tab to generate commented code.</div>
                  )}
                </>
              )}

              {view === 'flowchart' && (
                result.flowchart
                  ? <>
                    <div className="flow-hdr"><span>🔀 Control Flow Diagram</span><CopyButton text={result.flowchart} /></div>
                    <MermaidDiagram chart={sanitizeMermaid(result.flowchart)} />
                    <details style={{marginTop:8}}>
                      <summary style={{ cursor: 'pointer', color: '#888', fontSize: 12 }}>Raw Mermaid source</summary>
                      <pre className="codeblock">{sanitizeMermaid(result.flowchart)}</pre>
                    </details>
                  </>
                  : <div className="muted-c">No flowchart generated.</div>
              )}

            </div>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Debug Page — 4 modes ────────────────────────────────────────────────────────
function DebugPage() {
  const [mode, setMode] = useState<DebugMode>('analyze');
  const [stacktrace, setStacktrace] = useState('');
  const [errMsg, setErrMsg] = useState('');
  const [codeCtx, setCodeCtx] = useState('');
  const [logs, setLogs] = useState('');
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [ssName, setSsName] = useState('');
  const [sourceMap, setSourceMap] = useState('');
  // Custom code mode
  const [customCode, setCustomCode] = useState('');
  const [customLang, setCustomLang] = useState('python');
  const [customErr, setCustomErr] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState('');

  const run = async () => {
    setLoading(true); setErr(''); setResult(null);
    try {
      let r;
      if (mode === 'analyze') {
        r = await debugAnalyze({ stacktrace, error_message: errMsg, code_context: codeCtx, console_logs: logs });
      } else if (mode === 'multimodal') {
        r = await multimodalDebug({ stacktrace, console_logs: logs, screenshot_base64: screenshot || undefined });
      } else if (mode === 'decode') {
        r = await decodeStacktrace(stacktrace, sourceMap || undefined);
      } else {
        // custom — use code editor content as code_context + error message
        r = await debugAnalyze({
          stacktrace,
          error_message: customErr,
          code_context: customCode,
          console_logs: logs,
        });
      }
      if (r.data.error) setErr(r.data.error); else setResult(r.data);
    } catch (e: any) { setErr(e.response?.data?.detail || e.message); }
    finally { setLoading(false); }
  };

  const onSS = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return;
    setSsName(f.name);
    const rd = new FileReader();
    rd.onload = ev => setScreenshot((ev.target?.result as string).split(',')[1]);
    rd.readAsDataURL(f);
  };

  const cc = (c: string) => c === 'high' ? 'success' : c === 'medium' ? 'warning' : 'error';

  const canRun = mode === 'custom'
    ? (customCode.trim().length > 0 || customErr.trim().length > 0)
    : (stacktrace.trim().length > 0 || errMsg.trim().length > 0);

  return (
    <div className="split">
      <div className="split-left debug-left">
        <div className="split-hdr"><div className="split-title"><Bug size={15} /> Debug Input</div></div>
        <div className="dtabs">
          <button className={`dtab ${mode === 'analyze' ? 'active' : ''}`} onClick={() => { setMode('analyze'); setResult(null); }}><Terminal size={13} /> Analyze Error</button>
          <button className={`dtab ${mode === 'multimodal' ? 'active' : ''}`} onClick={() => { setMode('multimodal'); setResult(null); }}><Image size={13} /> Multimodal</button>
          <button className={`dtab ${mode === 'decode' ? 'active' : ''}`} onClick={() => { setMode('decode'); setResult(null); }}><Search size={13} /> Decode Stack</button>
          <button className={`dtab ${mode === 'custom' ? 'active' : ''}`} onClick={() => { setMode('custom'); setResult(null); }}><Code2 size={13} /> Custom Code</button>
        </div>

        {/* Custom Code Mode — Monaco editor */}
        {mode === 'custom' && (
          <div className="custom-debug-wrap">
            <div className="custom-debug-hdr">
              <span className="muted-c" style={{ fontSize: 12 }}>Paste your code + describe the error</span>
              <select className="sel" value={customLang} onChange={e => setCustomLang(e.target.value)}>
                {LANGS.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>
            <div className="custom-editor-box">
              <Editor
                height="100%"
                language={customLang}
                value={customCode}
                onChange={v => setCustomCode(v || '')}
                theme="vs-dark"
                options={{ fontSize: 12, minimap: { enabled: false }, wordWrap: 'on', scrollBeyondLastLine: false }}
              />
            </div>
            <input className="inp" placeholder="Error message / what's going wrong..." value={customErr} onChange={e => setCustomErr(e.target.value)} />
            <textarea className="ta" rows={2} placeholder="Stacktrace (optional)..." value={stacktrace} onChange={e => setStacktrace(e.target.value)} />
            <textarea className="ta" rows={2} placeholder="Console logs (optional)..." value={logs} onChange={e => setLogs(e.target.value)} />
          </div>
        )}

        {/* Standard modes */}
        {mode !== 'custom' && (
          <div className="dinputs">
            <textarea className="ta" rows={5} placeholder="Stacktrace / error trace..." value={stacktrace} onChange={e => setStacktrace(e.target.value)} />
            {mode === 'analyze' && <>
              <input className="inp" placeholder="Error message..." value={errMsg} onChange={e => setErrMsg(e.target.value)} />
              <textarea className="ta" rows={3} placeholder="Code context (optional)..." value={codeCtx} onChange={e => setCodeCtx(e.target.value)} />
              <textarea className="ta" rows={3} placeholder="Console logs (optional)..." value={logs} onChange={e => setLogs(e.target.value)} />
            </>}
            {mode === 'multimodal' && <>
              <textarea className="ta" rows={3} placeholder="Console logs..." value={logs} onChange={e => setLogs(e.target.value)} />
              <label className="upload-zone compact">
                <input type="file" accept="image/*" onChange={onSS} hidden />
                <Image size={18} /><span>{ssName || 'Upload screenshot (optional)'}</span>
              </label>
              {screenshot && <img src={`data:image/png;base64,${screenshot}`} alt="ss" className="ss-preview" />}
            </>}
            {mode === 'decode' && (
              <textarea className="ta" rows={4} placeholder="Source map content (optional)..." value={sourceMap} onChange={e => setSourceMap(e.target.value)} />
            )}
          </div>
        )}

        <button className="btn-pri full" onClick={run} disabled={loading || !canRun}>
          {loading ? <><Spinner /> Analyzing...</> : <><Zap size={14} /> Run Debug Analysis</>}
        </button>
      </div>

      <div className="split-right">
        {!result && !loading && !err && (
          <div className="empty">
            <Bug size={42} className="empty-ico" />
            <h3>AI Debugger</h3>
            <p>Paste your error — DevPilot finds the root cause, suggests a fix, and explains what went wrong.</p>
            <div className="feat-list">
              {[
                'Root cause analysis',
                'AI-generated fix with code',
                'Screenshot + logs + stacktrace correlation',
                'Minified stacktrace decoder',
                'Custom code editor debug',
              ].map(f => (
                <div key={f} className="feat"><CheckCircle2 size={13} /> {f}</div>
              ))}
            </div>
          </div>
        )}
        {loading && <div className="loading"><Spinner /><p>Running AI debug analysis...</p></div>}
        {err && <div className="err-box"><XCircle size={14} /> {err}</div>}

        {result && (mode === 'analyze' || mode === 'custom') && (
          <div className="rcontent">
            <div className="dbadges"><Badge text={result.error_type || 'Unknown'} type="warning" /><Badge text={`${result.confidence} confidence`} type={cc(result.confidence)} /></div>
            <div className="dcard red"><h4><XCircle size={13} /> Root Cause</h4><p>{result.root_cause}</p></div>
            <div className="dcard blue"><h4><Search size={13} /> Explanation</h4><p>{result.explanation}</p></div>
            <div className="dcard green">
              <h4><CheckCircle2 size={13} /> Fix</h4><p>{result.fix}</p>
              {result.fixed_code && <div className="codeblock-wrap"><CopyButton text={result.fixed_code} /><pre className="codeblock">{result.fixed_code}</pre></div>}
            </div>
            {result.prevention && <div className="dcard purple"><h4><Sparkles size={13} /> Prevention</h4><p>{result.prevention}</p></div>}
            {result.related_issues?.length > 0 && (
              <Collapsible title="Related Issues" icon={<AlertTriangle size={13} />}>
                <ul className="issue-list">{result.related_issues.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
              </Collapsible>
            )}
          </div>
        )}

        {result && mode === 'multimodal' && (
          <div className="rcontent">
            {result.screenshot_interpretation && <div className="dcard blue"><h4><Image size={13} /> Screenshot</h4><p>{result.screenshot_interpretation}</p></div>}
            <div className="dcard red"><h4><Bug size={13} /> Likely Location</h4><p><b>File:</b> {result.most_likely_file}</p><p><b>Line:</b> {result.most_likely_line}</p></div>
            <div className="dcard orange"><h4><Layers size={13} /> Signal Correlation</h4><p>{result.signal_correlation}</p></div>
            <div className="dcard green"><h4><CheckCircle2 size={13} /> Root Cause</h4><p>{result.root_cause}</p></div>
            {result.fix_steps?.length > 0 && <div className="dcard purple"><h4>Fix Steps</h4><ol>{result.fix_steps.map((s: string, i: number) => <li key={i}>{s}</li>)}</ol></div>}
          </div>
        )}

        {result && mode === 'decode' && (
          <div className="rcontent">
            <div className="dcard blue"><h4><Search size={13} /> Summary</h4><p>{result.summary}</p></div>
            <div className="dcard purple"><h4><FileText size={13} /> Source Files</h4><ul>{result.likely_source_files?.map((f: string, i: number) => <li key={i}><code>{f}</code></li>)}</ul></div>
            <div className="dcard green"><h4><Terminal size={13} /> Entry Point</h4><p>{result.entry_point}</p></div>
            {result.decoded_frames?.length > 0 && (
              <Collapsible title={`Decoded Frames (${result.decoded_frames.length})`} defaultOpen icon={<Code2 size={13} />}>
                {result.decoded_frames.map((f: any, i: number) => (
                  <div key={i} className="frame-row">
                    <Badge text={f.confidence} type={cc(f.confidence)} />
                    <div><div className="frame-orig">{f.original}</div><div className="frame-dec"><ChevronRight size={11} /> {f.decoded}</div></div>
                  </div>
                ))}
              </Collapsible>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Chat Page ───────────────────────────────────────────────────────────────────
function ChatPage({ sessionId, indexedCount }: { sessionId: string; indexedCount: number }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const msg: ChatMessage = { role: 'user', content: input.trim(), timestamp: Date.now() };
    const hist = messages.map(m => ({ role: m.role, content: m.content }));
    setMessages(p => [...p, msg]);
    setInput('');
    setLoading(true);
    try {
      const r = await sendChat(msg.content, sessionId, hist);
      setMessages(p => [...p, { role: 'assistant', content: r.data.response, timestamp: Date.now() }]);
    } catch (e: any) {
      setMessages(p => [...p, { role: 'assistant', content: `Error: ${e.message}`, timestamp: Date.now() }]);
    } finally { setLoading(false); }
  };

  const SUGGESTIONS = [
    'Explain how authentication works',
    'What handles database queries?',
    'Find potential security issues',
    'How is error handling implemented?',
    'What are the main API endpoints?',
  ];

  return (
    <div className="chat-layout">
      <div className={`chat-banner ${indexedCount > 0 ? 'ok' : ''}`}>
        {indexedCount > 0
          ? <><CheckCircle2 size={14} /> <b>{indexedCount}</b> files indexed — chat has full codebase context</>
          : <><AlertTriangle size={14} /> Connect a repo or upload files for context-aware chat</>}
      </div>
      <div className="chat-msgs">
        {messages.length === 0 && (
          <div className="chat-empty">
            <MessageSquare size={38} className="empty-ico" />
            <h3>Ask anything about your code</h3>
            <div className="suggestions">
              {SUGGESTIONS.map(s => <button key={s} className="sugg" onClick={() => setInput(s)}>{s}</button>)}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="msg-av">{m.role === 'user' ? 'U' : 'AI'}</div>
            <div className="msg-bubble">
              {m.role === 'assistant' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, className, children, ...props }: any) {
                      const lang = (className || '').replace('language-', '').toLowerCase();
                      const code = String(children).replace(/\n$/, '').trim();
                      // Render as Mermaid if: lang is mermaid/graph/flowchart/sequence,
                      // OR code body starts with a known diagram keyword
                      const MERMAID_LANGS = ['mermaid','graph','flowchart','sequencediagram','sequencediagram','classdiagram','statediagram','erdiagram','gantt','pie','gitgraph'];
                      const MERMAID_STARTS = /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph)/i;
                      const isMermaid = MERMAID_LANGS.includes(lang) || (!lang && MERMAID_STARTS.test(code));
                      if (isMermaid) return <MermaidDiagram chart={sanitizeMermaid(code)} />;
                      return <code className={className} {...props}>{children}</code>;
                    }
                  }}
                >{m.content}</ReactMarkdown>
              ) : <p>{m.content}</p>}
            </div>
          </div>
        ))}
        {loading && (
          <div className="msg assistant">
            <div className="msg-av">AI</div>
            <div className="msg-bubble typing"><span /><span /><span /></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <textarea className="chat-inp" rows={2} placeholder="Ask about your code... (Enter to send)" value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} />
        <button className="btn-pri send-btn" onClick={send} disabled={loading || !input.trim()}>
          {loading ? <Spinner /> : <ChevronRight size={18} />}
        </button>
      </div>
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState<Tab>('docs');
  const [sessionId, setSessionId] = useState('');
  const [indexedCount, setIndexedCount] = useState(0);
  const [indexedFiles, setIndexedFiles] = useState<IndexedFile[]>([]);
  const [lastRepoInfo, setLastRepoInfo] = useState<{owner:string,repo:string,branch:string} | null>(null);
  const [ingestOpen, setIngestOpen] = useState(false);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    createSession().then(r => setSessionId(r.data.session_id)).catch(() => setSessionId('local-' + Date.now()));
    // Retry health check every 5s until connected — Render cold-start can take 30-60s
    let cancelled = false;
    const tryConnect = async (attempt: number) => {
      try {
        await checkHealth();
        if (!cancelled) setApiOk(true);
      } catch {
        if (!cancelled) {
          // Keep retrying silently for up to 2 minutes (24 attempts × 5s)
          if (attempt < 24) {
            setTimeout(() => tryConnect(attempt + 1), 5000);
          } else {
            setApiOk(false); // Only mark offline after 2 min of retrying
          }
        }
      }
    };
    tryConnect(0);
    return () => { cancelled = true; };
  }, []);

  const handleIndexed = (n: number, files: IndexedFile[], repoInfo?: {owner:string,repo:string,branch:string}) => {
    setIndexedCount(p => p + n);
    if (repoInfo) setLastRepoInfo(repoInfo);
    setIndexedFiles(prev => {
      const existing = new Set(prev.map(f => f.filepath));
      const newFiles = files.filter(f => !existing.has(f.filepath));
      return [...prev, ...newFiles];
    });
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo"><Zap size={20} className="logo-ico" /><span>DevPilot</span></div>
        <nav>
          <button className={`nav-item ${tab === 'docs' ? 'active' : ''}`} onClick={() => setTab('docs')}><FileText size={17} /><span>Documentation</span></button>
          <button className={`nav-item ${tab === 'debug' ? 'active' : ''}`} onClick={() => setTab('debug')}><Bug size={17} /><span>Debugger</span></button>
          <button className={`nav-item ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}><MessageSquare size={17} /><span>Code Chat</span>{indexedCount > 0 && <span className="nbadge">{indexedCount}</span>}</button>
        </nav>
        <div className="sidebar-foot">
          <div className={`api-dot ${apiOk === true ? 'ok' : apiOk === false ? 'err' : ''}`} />
          <span>{apiOk === true ? 'API Connected' : apiOk === false ? 'API Unreachable' : 'Server waking up...'}</span>
        </div>
      </aside>

      <main className="main">
        <div className="main-hdr">
          <div className="main-hdr-top">
            <div className="main-title">
              {tab === 'docs' && <><FileText size={18} /> Documentation Helper</>}
              {tab === 'debug' && <><Bug size={18} /> AI Debugger</>}
              {tab === 'chat' && <><MessageSquare size={18} /> Code Chat</>}
            </div>
              <button
                className={`ingest-toggle ${ingestOpen ? 'open' : ''}`}
                onClick={() => setIngestOpen((o: boolean) => !o)}
                title="Toggle repo indexing panel"
              >
                <GitBranch size={13} />
                {indexedCount > 0
                  ? <span className="ingest-toggle-label">{indexedCount} files indexed</span>
                  : <span className="ingest-toggle-label">Index Repo</span>}
                {apiOk === null && <span className="ingest-toggle-api-badge connecting">Waking up...</span>}
                {apiOk === false && <span className="ingest-toggle-api-badge offline">Unreachable</span>}
                <ChevronDown size={12} className="ingest-chevron" />
              </button>
          </div>
          {ingestOpen && (
            <div className="ingest-dropdown">
              {/* API not yet connected — show banner inside dropdown */}
              {apiOk !== true && (
                <div className={`api-wait-banner ${apiOk === false ? 'err' : ''}`}>
                  {apiOk === null
                    ? <><Spinner /> <span><b>Backend is waking up</b> — our server (Render free tier) takes 20–30s to start after being idle. Retrying automatically. You can type your repo URL now and click Index Repo once connected.</span></>
                    : <><XCircle size={13} /> <span><b>Could not reach the backend</b> after 2 minutes. Please refresh the page to try again.</span></>
                  }
                </div>
              )}
              {sessionId && (
                <IngestionPanel sessionId={sessionId} apiOk={apiOk} onIndexed={(n: number, files: IndexedFile[], ri?: any) => { handleIndexed(n, files, ri); if (n > 0) setIngestOpen(false); }} />
              )}
            </div>
          )}
        </div>
        <div className="main-body">
          {tab === 'docs' && <DocsPage sessionId={sessionId} indexedFiles={indexedFiles} indexedCount={indexedCount} repoInfo={lastRepoInfo} />}
          {tab === 'debug' && <DebugPage />}
          {tab === 'chat' && <ChatPage sessionId={sessionId} indexedCount={indexedCount} />}
        </div>
      </main>
    </div>
  );
}
