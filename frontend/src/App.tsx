import { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Github, Upload, MessageSquare, Bug, FileText, ChevronRight,
  Zap, Copy, Check, AlertTriangle, CheckCircle2,
  XCircle, Loader2, Image, Code2, GitBranch, Layers, Search,
  Terminal, Sparkles, ChevronDown, ChevronUp,
} from 'lucide-react';
import mermaid from 'mermaid';
import {
  createSession, ingestGithub, ingestFiles, explainCode,
  debugAnalyze, multimodalDebug, decodeStacktrace, sendChat, checkHealth,
} from './lib/api';
import type { ExplainResult, ChatMessage, Tab, DebugMode } from './types';
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
    const id = `mermaid-${Date.now()}`;
    mermaid.render(id, chart)
      .then(({ svg }) => { if (ref.current) ref.current.innerHTML = svg; })
      .catch(() => { if (ref.current) ref.current.innerHTML = `<pre style="color:#aaa;font-size:12px;padding:12px">${chart}</pre>`; });
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

// ── Ingestion Panel ────────────────────────────────────────────────────────────
function IngestionPanel({ sessionId, onIndexed }: { sessionId: string; onIndexed: (n: number) => void }) {
  const [tab, setTab] = useState<'gh' | 'up'>('gh');
  const [url, setUrl] = useState('');
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState('');

  const doGithub = async () => {
    if (!url.trim()) return;
    setLoading(true); setErr(''); setRes(null);
    try {
      const r = await ingestGithub(url.trim(), sessionId, token || undefined);
      setRes(r.data); onIndexed(r.data.indexed_files);
    } catch (e: any) { setErr(e.response?.data?.detail || e.message); }
    finally { setLoading(false); }
  };

  const doUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setLoading(true); setErr(''); setRes(null);
    try {
      const data = await Promise.all(files.map(async f => ({ filename: f.name, content: await f.text() })));
      const r = await ingestFiles(data, sessionId);
      setRes(r.data); onIndexed(r.data.indexed_files);
    } catch (e: any) { setErr(e.response?.data?.detail || e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="ingest">
      <div className="ingest-tabs">
        <button className={`itab ${tab === 'gh' ? 'active' : ''}`} onClick={() => setTab('gh')}><Github size={13} /> GitHub</button>
        <button className={`itab ${tab === 'up' ? 'active' : ''}`} onClick={() => setTab('up')}><Upload size={13} /> Upload</button>
      </div>
      {tab === 'gh' && (
        <div className="ingest-row">
          <input className="inp" placeholder="https://github.com/owner/repo" value={url} onChange={e => setUrl(e.target.value)} onKeyDown={e => e.key === 'Enter' && doGithub()} />
          <input className="inp" type="password" placeholder="GitHub Token (optional)" value={token} onChange={e => setToken(e.target.value)} />
          <button className="btn-pri" onClick={doGithub} disabled={loading || !url}>{loading ? <Spinner /> : <GitBranch size={14} />} Index Repo</button>
        </div>
      )}
      {tab === 'up' && (
        <label className="upload-zone">
          <input type="file" multiple accept=".py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.cpp,.c,.md,.txt,.json" onChange={doUpload} hidden />
          <Upload size={22} /> <span>Click to upload code files</span>
        </label>
      )}
      {loading && <div className="status-row"><Spinner /> Indexing... (may take 30–60s on first run while server warms up)</div>}
      {err && <div className="err-row"><XCircle size={13} /> {err}</div>}
      {res && <div className="ok-row"><CheckCircle2 size={13} /> Indexed <b>{res.indexed_files}</b> files{res.repo ? ` from ${res.repo}` : ''}</div>}
    </div>
  );
}

// ── Docs Page ──────────────────────────────────────────────────────────────────
function DocsPage() {
  const [code, setCode] = useState(`def calculate_statistics(numbers):
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    variance = sum((x - mean) ** 2 for x in numbers) / count
    std_dev = variance ** 0.5
    return {"mean": mean, "std_dev": std_dev, "variance": variance}

def find_outliers(data, threshold=2.0):
    stats = calculate_statistics(data)
    mean, std = stats["mean"], stats["std_dev"]
    return [x for x in data if abs(x - mean) > threshold * std]`);
  const [lang, setLang] = useState('python');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [err, setErr] = useState('');
  const [view, setView] = useState<'overview' | 'functions' | 'flowchart'>('overview');

  const analyze = async () => {
    setLoading(true); setErr(''); setResult(null);
    try {
      const r = await explainCode(code, lang);
      if (r.data.error) setErr(r.data.error);
      else { setResult(r.data); setView('overview'); }
    } catch (e: any) { setErr(e.response?.data?.detail || e.message || 'Error'); }
    finally { setLoading(false); }
  };

  return (
    <div className="split">
      <div className="split-left">
        <div className="split-hdr">
          <div className="split-title"><Code2 size={15} /> Code Input</div>
          <select className="sel" value={lang} onChange={e => setLang(e.target.value)}>
            {LANGS.map(l => <option key={l}>{l}</option>)}
          </select>
        </div>
        <div className="editor-box">
          <Editor height="100%" language={lang} value={code} onChange={v => setCode(v || '')}
            theme="vs-dark" options={{ fontSize: 13, minimap: { enabled: false }, wordWrap: 'on', scrollBeyondLastLine: false }} />
        </div>
        <button className="btn-pri full" onClick={analyze} disabled={loading}>
          {loading ? <><Spinner /> Analyzing...</> : <><Sparkles size={14} /> Analyze & Document</>}
        </button>
      </div>

      <div className="split-right">
        {!result && !loading && !err && (
          <div className="empty">
            <Sparkles size={42} className="empty-ico" />
            <h3>Documentation Helper</h3>
            <p>Paste messy code on the left. DevPilot explains every function, generates docstrings, and draws a flowchart.</p>
            <div className="feat-list">
              {['Plain-English overview', 'Function-by-function breakdown', 'Auto docstrings', 'Mermaid flowchart'].map(f => (
                <div key={f} className="feat"><CheckCircle2 size={13} /> {f}</div>
              ))}
            </div>
          </div>
        )}
        {loading && <div className="loading"><Spinner /><p>Analyzing with AI...</p></div>}
        {err && <div className="err-box"><XCircle size={14} /> {err}</div>}
        {result && (
          <div className="result">
            <div className="rtabs">
              <button className={`rtab ${view === 'overview' ? 'active' : ''}`} onClick={() => setView('overview')}><FileText size={12} /> Overview</button>
              <button className={`rtab ${view === 'functions' ? 'active' : ''}`} onClick={() => setView('functions')}><Code2 size={12} /> Functions ({result.functions?.length || 0})</button>
              <button className={`rtab ${view === 'flowchart' ? 'active' : ''}`} onClick={() => setView('flowchart')}><GitBranch size={12} /> Flowchart</button>
            </div>

            <div className="rcontent">
              {view === 'overview' && (
                <>
                  <div className="card blue-card"><h4>What it does</h4><p>{result.overview}</p></div>
                  {result.key_concepts?.length > 0 && <div className="tags">{result.key_concepts.map(c => <Badge key={c} text={c} />)}</div>}
                  {result.complexity && <div className="info-card"><b>Complexity:</b> {result.complexity}</div>}
                  {result.potential_issues?.length > 0 && (
                    <div className="warn-card">
                      <b><AlertTriangle size={13} /> Potential Issues</b>
                      <ul>{result.potential_issues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                    </div>
                  )}
                </>
              )}
              {view === 'functions' && result.functions?.map((fn, i) => (
                <Collapsible key={i} title={fn.name} defaultOpen={i === 0} icon={<Code2 size={13} />}>
                  <div className="fn-rows">
                    <div><b>Purpose:</b> {fn.purpose}</div>
                    {fn.parameters && <div><b>Parameters:</b> {fn.parameters}</div>}
                    {fn.returns && <div><b>Returns:</b> {fn.returns}</div>}
                    <div><b>Logic:</b> {fn.logic}</div>
                    {fn.docstring && (
                      <div className="docblock">
                        <div className="docblock-hdr"><b>Docstring</b><CopyButton text={fn.docstring} /></div>
                        <pre className="codeblock">{fn.docstring}</pre>
                      </div>
                    )}
                  </div>
                </Collapsible>
              ))}
              {view === 'flowchart' && (result.flowchart
                ? <>
                  <div className="flow-hdr"><span>Mermaid Diagram</span><CopyButton text={result.flowchart} /></div>
                  <MermaidDiagram chart={result.flowchart} />
                  <details><summary style={{ cursor: 'pointer', color: '#888', fontSize: 12 }}>Raw Mermaid</summary><pre className="codeblock">{result.flowchart}</pre></details>
                </>
                : <div className="muted-c">No flowchart generated</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Debug Page ─────────────────────────────────────────────────────────────────
function DebugPage() {
  const [mode, setMode] = useState<DebugMode>('analyze');
  const [stacktrace, setStacktrace] = useState('');
  const [errMsg, setErrMsg] = useState('');
  const [codeCtx, setCodeCtx] = useState('');
  const [logs, setLogs] = useState('');
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [ssName, setSsName] = useState('');
  const [sourceMap, setSourceMap] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState('');

  const run = async () => {
    setLoading(true); setErr(''); setResult(null);
    try {
      let r;
      if (mode === 'analyze') r = await debugAnalyze({ stacktrace, error_message: errMsg, code_context: codeCtx, console_logs: logs });
      else if (mode === 'multimodal') r = await multimodalDebug({ stacktrace, console_logs: logs, screenshot_base64: screenshot || undefined });
      else r = await decodeStacktrace(stacktrace, sourceMap || undefined);
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

  return (
    <div className="split">
      <div className="split-left debug-left">
        <div className="split-hdr"><div className="split-title"><Bug size={15} /> Debug Input</div></div>
        <div className="dtabs">
          <button className={`dtab ${mode === 'analyze' ? 'active' : ''}`} onClick={() => setMode('analyze')}><Terminal size={13} /> Analyze Error</button>
          <button className={`dtab ${mode === 'multimodal' ? 'active' : ''}`} onClick={() => setMode('multimodal')}><Image size={13} /> Multimodal</button>
          <button className={`dtab ${mode === 'decode' ? 'active' : ''}`} onClick={() => setMode('decode')}><Search size={13} /> Decode Stack</button>
        </div>
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
        <button className="btn-pri full" onClick={run} disabled={loading || (!stacktrace && !errMsg)}>
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
              {['Root cause analysis', 'AI-generated fix', 'Screenshot + logs + stacktrace correlation', 'Minified stacktrace decoder'].map(f => (
                <div key={f} className="feat"><CheckCircle2 size={13} /> {f}</div>
              ))}
            </div>
          </div>
        )}
        {loading && <div className="loading"><Spinner /><p>Running AI debug analysis...</p></div>}
        {err && <div className="err-box"><XCircle size={14} /> {err}</div>}

        {result && mode === 'analyze' && (
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

// ── Chat Page ──────────────────────────────────────────────────────────────────
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
                      const lang = (className || '').replace('language-', '');
                      const code = String(children).replace(/\n$/, '');
                      if (lang === 'mermaid') {
                        return <MermaidDiagram chart={code} />;
                      }
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

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState<Tab>('docs');
  const [sessionId, setSessionId] = useState('');
  const [indexedCount, setIndexedCount] = useState(0);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    createSession().then(r => setSessionId(r.data.session_id)).catch(() => setSessionId('local-' + Date.now()));
    checkHealth().then(() => setApiOk(true)).catch(() => setApiOk(false));
  }, []);

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
          <span>{apiOk === true ? 'API Connected' : apiOk === false ? 'API Offline' : 'Connecting...'}</span>
        </div>
      </aside>

      <main className="main">
        <div className="main-hdr">
          <div className="main-title">
            {tab === 'docs' && <><FileText size={18} /> Documentation Helper</>}
            {tab === 'debug' && <><Bug size={18} /> AI Debugger</>}
            {tab === 'chat' && <><MessageSquare size={18} /> Code Chat</>}
          </div>
          {sessionId && (tab === 'docs' || tab === 'chat') && (
            <IngestionPanel sessionId={sessionId} onIndexed={n => setIndexedCount(p => p + n)} />
          )}
        </div>
        <div className="main-body">
          {tab === 'docs' && <DocsPage />}
          {tab === 'debug' && <DebugPage />}
          {tab === 'chat' && <ChatPage sessionId={sessionId} indexedCount={indexedCount} />}
        </div>
      </main>
    </div>
  );
}
