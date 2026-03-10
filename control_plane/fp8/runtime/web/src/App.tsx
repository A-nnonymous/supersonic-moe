import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchState, launchWorkers, saveConfig, stopAll, stopWorkers } from './api';
import type {
  ConfigShape,
  ConfigWorker,
  DashboardState,
  GateItem,
  HeartbeatAgent,
  MergeQueueItem,
  ProcessSnapshot,
  ProviderQueueItem,
  RuntimeWorker,
  TabKey,
} from './types';

const AUTO_REFRESH_MS = 4000;

function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ');
}

function stateClass(value: string | undefined): string {
  return `state-${String(value || 'unknown').replace(/[^a-zA-Z0-9]+/g, '_')}`;
}

function displayState(value: string | undefined): string {
  return String(value || 'unknown').replaceAll('_', ' ');
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return ' ';
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  return String(value);
}

type AgentRow = {
  agent: string;
  role: string;
  provider: string;
  model: string;
  resource_pool: string;
  branch: string;
  heartbeat_state?: string;
  runtime_status?: string;
  process_alive?: boolean;
  pid?: number;
  evidence?: string;
  expected_next_checkin?: string;
  last_seen?: string;
  display_state: string;
};

function sortAgents(rows: AgentRow[]): AgentRow[] {
  return [...rows].sort((left, right) => {
    const leftNum = Number(String(left.agent || '').replace(/[^0-9]/g, ''));
    const rightNum = Number(String(right.agent || '').replace(/[^0-9]/g, ''));
    return leftNum - rightNum;
  });
}

function buildAgentRows(data: DashboardState | null): AgentRow[] {
  if (!data) {
    return [];
  }
  const byAgent = new Map<string, Partial<AgentRow>>();
  const remember = (agent: string | undefined, values: Partial<AgentRow>) => {
    if (!agent) {
      return;
    }
    byAgent.set(agent, { ...(byAgent.get(agent) || { agent }), ...values, agent });
  };

  (data.runtime?.workers || []).forEach((item: RuntimeWorker) => {
    remember(item.agent, {
      provider: item.provider,
      model: item.model,
      resource_pool: item.resource_pool,
      branch: item.branch,
      runtime_status: item.status,
    });
  });

  (data.config?.workers || []).forEach((item: ConfigWorker) => {
    remember(item.agent, {
      branch: item.branch,
    });
  });

  (data.heartbeats?.agents || []).forEach((item: HeartbeatAgent) => {
    remember(item.agent, {
      role: item.role,
      heartbeat_state: item.state,
      last_seen: item.last_seen,
      evidence: item.evidence,
      expected_next_checkin: item.expected_next_checkin,
    });
  });

  Object.entries(data.processes || {}).forEach(([agent, item]: [string, ProcessSnapshot]) => {
    remember(agent, {
      process_alive: item.alive,
      pid: item.pid,
      provider: item.provider,
      model: item.model,
      resource_pool: item.resource_pool,
    });
  });

  return sortAgents(
    Array.from(byAgent.values()).map((item) => {
      const state = item.process_alive ? 'active' : item.heartbeat_state || item.runtime_status || 'not_started';
      return {
        agent: item.agent || 'unknown',
        role: item.role || 'worker',
        provider: item.provider || 'unassigned',
        model: item.model || 'unassigned',
        resource_pool: item.resource_pool || 'unassigned',
        branch: item.branch || 'unassigned',
        heartbeat_state: item.heartbeat_state,
        runtime_status: item.runtime_status,
        process_alive: item.process_alive,
        pid: item.pid,
        evidence: item.evidence,
        expected_next_checkin: item.expected_next_checkin,
        last_seen: item.last_seen,
        display_state: state,
      };
    }),
  );
}

type ProgressModel = {
  progress: number;
  passedGates: number;
  totalGates: number;
  completedItems: number;
  totalItems: number;
  blockedItems: number;
  activeAgents: number;
  attentionAgents: number;
  openGate?: GateItem;
};

function buildProgressModel(data: DashboardState | null, agentRows: AgentRow[]): ProgressModel {
  const gates = data?.gates?.gates || [];
  const backlog = data?.backlog?.items || [];
  const passedGates = gates.filter((item) => item.status === 'passed').length;
  const progress = gates.length ? Math.round((passedGates / gates.length) * 100) : 0;
  const completedItems = backlog.filter((item) => ['done', 'completed', 'closed'].includes(String(item.status))).length;
  const blockedItems = backlog.filter((item) => String(item.status) === 'blocked').length;
  const activeAgents = agentRows.filter((item) => item.display_state === 'active' || item.display_state === 'healthy').length;
  const attentionAgents = agentRows.filter((item) => item.display_state === 'stale' || item.display_state.startsWith('launch_failed')).length;
  const openGate = gates.find((item) => item.status !== 'passed');
  return { progress, passedGates, totalGates: gates.length, completedItems, totalItems: backlog.length, blockedItems, activeAgents, attentionAgents, openGate };
}

function DataTable({ columns, rows }: { columns: string[]; rows: Array<Record<string, unknown>> }) {
  if (!rows.length) {
    return <div className="small muted">No data</div>;
  }
  return (
    <table>
      <thead>
        <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {columns.map((column) => <td key={column}>{renderCell(row[column])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OverviewTab({ data, agentRows, progress }: { data: DashboardState; agentRows: AgentRow[]; progress: ProgressModel }) {
  const mergeQueue = data.merge_queue || [];
  const mergeReady = mergeQueue.filter((item) => ['offline', 'stopped'].includes(String(item.status))).length;
  const mergeActive = mergeQueue.filter((item) => ['active', 'healthy'].includes(String(item.status))).length;
  return (
    <div className="tab-body">
      <section className="overview-hero">
        <section className="card progress-card">
          <div className="page-header">
            <div>
              <h2>Overall Progress</h2>
              <p className="small">A compact view of delivery momentum and the current control-plane state.</p>
            </div>
            <div className="small muted">{progress.passedGates}/{progress.totalGates} gates passed</div>
          </div>
          <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress.progress}%` }} /></div>
          <div className="summary">
            <Metric label="Agents" value={agentRows.length} hint={`${progress.activeAgents} active or healthy`} />
            <Metric label="Overall Progress" value={`${progress.progress}%`} hint={`${progress.passedGates}/${progress.totalGates} gates passed`} />
            <Metric label="Attention Needed" value={progress.attentionAgents} hint={`${progress.blockedItems} backlog items blocked`} />
            <Metric label="Launch Blockers" value={data.launch_blockers.length} hint={data.launch_blockers.length ? `${data.validation_errors.length} config notes` : 'launch path is ready'} />
          </div>
          <div className="progress-list">
            <ProgressRow label="Backlog" value={`${progress.completedItems}/${progress.totalItems} completed`} />
            <ProgressRow label="Blocked work" value={`${progress.blockedItems} items`} />
            <ProgressRow label="Agents needing action" value={`${progress.attentionAgents}`} />
            <ProgressRow label="Current gate" value={progress.openGate ? `${progress.openGate.id} · ${progress.openGate.name}` : 'All gates passed'} />
          </div>
        </section>
        <section className="card">
          <div className="page-header">
            <div>
              <h2>Program Snapshot</h2>
              <p className="small">What is blocked, what is runnable, and which event happened last.</p>
            </div>
          </div>
          <div className="helper-list">
            <HelperCard title="Startup state" body={data.mode.reason || data.mode.state} />
            <HelperCard title="Config target" body={data.mode.persist_config_path} />
            <HelperCard title="Last event" body={data.last_event || 'none'} />
            <HelperCard title="Launch posture" body={data.launch_blockers.length ? `${data.launch_blockers.length} blocker(s)` : 'ready to launch'} />
          </div>
        </section>
      </section>

      <section className="card">
        <div className="panel-title">
          <div>
            <h2>Branch Merge Status</h2>
            <p className="small">Manager-owned merge visibility for every worker branch.</p>
          </div>
          <div className="small muted">{mergeActive} in progress, {mergeReady} ready for review</div>
        </div>
        <div className="merge-board">
          {mergeQueue.length ? mergeQueue.map((item) => <MergeCard key={`${item.agent}-${item.branch}`} item={item} />) : <div className="small muted">No worker branches registered for manager merge review.</div>}
        </div>
      </section>

      <section className="card">
        <div className="panel-title">
          <div>
            <h2>Agent Dashboards</h2>
            <p className="small">Health, execution context, and current ownership.</p>
          </div>
          <div className="small muted">{progress.activeAgents} active, {progress.attentionAgents} need attention</div>
        </div>
        <div className="agent-wall">
          {agentRows.map((item) => <AgentCard key={item.agent} item={item} />)}
        </div>
      </section>
    </div>
  );
}

function OperationsTab({ data }: { data: DashboardState }) {
  const projectRows = [
    { key: 'repository_name', value: data.project.repository_name || '' },
    { key: 'local_repo_root', value: data.project.local_repo_root || '' },
    { key: 'paddle_repo_path', value: data.project.paddle_repo_path || '' },
    { key: 'integration_branch', value: data.project.integration_branch || data.project.base_branch || '' },
    { key: 'dashboard', value: data.project.dashboard?.host && data.project.dashboard?.port ? `${data.project.dashboard.host}:${data.project.dashboard.port}` : '' },
  ];
  const processRows = Object.entries(data.processes || {}).map(([agent, item]) => ({ agent, provider: item.provider, model: item.model, alive: item.alive, pid: item.pid, resource_pool: item.resource_pool, returncode: item.returncode }));
  const mergeRows = data.merge_queue.map((item) => ({ agent: item.agent, branch: item.branch, submit_strategy: item.submit_strategy, worker_identity: item.worker_identity, merge_target: item.merge_target, status: item.status, manager_action: item.manager_action }));
  const providerRows = data.provider_queue.map((item) => ({ resource_pool: item.resource_pool, provider: item.provider, priority: item.priority, binary_found: item.binary_found, api_key_present: item.api_key_present, connection_quality: item.connection_quality, work_quality: item.work_quality, score: item.score }));
  return (
    <div className="tab-body">
      <section className="grid">
        <section className="card"><h2>Commands</h2><pre>{`serve:\n${data.commands.serve}\n\nup:\n${data.commands.up}`}</pre></section>
        <section className="card"><h2>Validation</h2><pre>{renderValidation(data)}</pre></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Provider Queue</h2><DataTable columns={['resource_pool', 'provider', 'priority', 'binary_found', 'api_key_present', 'connection_quality', 'work_quality', 'score']} rows={providerRows} /></section>
        <section className="card"><h2>Merge Queue</h2><DataTable columns={['agent', 'branch', 'submit_strategy', 'worker_identity', 'merge_target', 'status', 'manager_action']} rows={mergeRows} /></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Active Processes</h2><DataTable columns={['agent', 'provider', 'model', 'alive', 'pid', 'resource_pool', 'returncode']} rows={processRows} /></section>
        <section className="card"><h2>Project</h2><DataTable columns={['key', 'value']} rows={projectRows} /></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Runtime Topology</h2><DataTable columns={['agent', 'resource_pool', 'provider', 'model', 'branch', 'status']} rows={data.runtime.workers || []} /></section>
        <section className="card"><h2>Heartbeats</h2><DataTable columns={['agent', 'state', 'last_seen', 'expected_next_checkin']} rows={data.heartbeats.agents || []} /></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Backlog</h2><DataTable columns={['id', 'owner', 'status', 'gate', 'title']} rows={data.backlog.items || []} /></section>
        <section className="card"><h2>Gates</h2><DataTable columns={['id', 'name', 'status', 'owner']} rows={data.gates.gates || []} /></section>
      </section>
      <section className="card"><h2>Manager Report</h2><pre>{data.manager_report}</pre></section>
    </div>
  );
}

function SettingsTab({ data, configText, onChange, onSave }: { data: DashboardState; configText: string; onChange: (value: string) => void; onSave: () => void }) {
  const pools = Object.entries(data.config.resource_pools || {}).map(([name, item]) => ({ name, priority: item.priority ?? 100, provider: item.provider, model: item.model }));
  const workers = (data.config.workers || []).map((item) => ({ agent: item.agent, task_id: item.task_id, resource_pool: item.resource_pool, resource_pool_queue: (item.resource_pool_queue || []).join(', '), branch: item.branch, git_identity: item.git_identity ? `${item.git_identity.name || ''} <${item.git_identity.email || ''}>` : 'environment default', submit_strategy: item.submit_strategy, test_command: item.test_command }));
  const manager = data.project.manager_git_identity ? `${data.project.manager_git_identity.name || ''} <${data.project.manager_git_identity.email || ''}>` : 'A0 manager identity';
  const mergePolicy = [
    { key: 'integration_branch', value: data.project.integration_branch || data.project.base_branch || 'main' },
    { key: 'manager_identity', value: manager },
    { key: 'merge_owner', value: 'A0' },
    { key: 'tracked_worker_branches', value: String(data.merge_queue.length) },
  ];
  const projectRows = [
    { key: 'repository_name', value: data.project.repository_name || '' },
    { key: 'local_repo_root', value: data.project.local_repo_root || '' },
    { key: 'paddle_repo_path', value: data.project.paddle_repo_path || '' },
    { key: 'integration_branch', value: data.project.integration_branch || data.project.base_branch || '' },
    { key: 'dashboard', value: data.project.dashboard?.host && data.project.dashboard?.port ? `${data.project.dashboard.host}:${data.project.dashboard.port}` : '' },
  ];
  return (
    <div className="tab-body">
      <section className="card">
        <div className="page-header">
          <div>
            <h2>Settings</h2>
            <p className="small">Edit API keys, provider routing, worktrees, Paddle path, and worker commands here.</p>
          </div>
          <button onClick={onSave}>Save Settings</button>
        </div>
        <div className="config-layout">
          <div>
            <textarea value={configText} onChange={(event) => onChange(event.target.value)} />
          </div>
          <div className="helper-list">
            <section className="helper-card"><h3>Project</h3><DataTable columns={['key', 'value']} rows={projectRows} /></section>
            <section className="helper-card"><h3>Resource Pools</h3><DataTable columns={['name', 'priority', 'provider', 'model']} rows={pools} /></section>
            <section className="helper-card"><h3>Merge Policy</h3><DataTable columns={['key', 'value']} rows={mergePolicy} /></section>
            <section className="helper-card"><h3>Worker Config</h3><DataTable columns={['agent', 'task_id', 'resource_pool', 'resource_pool_queue', 'branch', 'git_identity', 'submit_strategy', 'test_command']} rows={workers} /></section>
          </div>
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string | number; hint: string }) {
  return <div className="metric"><strong>{value}</strong><div>{label}</div><div className="small">{hint}</div></div>;
}

function ProgressRow({ label, value }: { label: string; value: string }) {
  return <div className="progress-row"><span className="small">{label}</span><strong>{value}</strong></div>;
}

function HelperCard({ title, body }: { title: string; body: string }) {
  return <section className="helper-card"><h3>{title}</h3><p className="small">{body}</p></section>;
}

function MergeCard({ item }: { item: MergeQueueItem }) {
  const raw = String(item.status || 'not_started');
  const status = raw === 'active' || raw === 'healthy'
    ? { label: 'In progress', className: 'state-active' }
    : raw === 'stale' || raw.startsWith('launch_failed')
      ? { label: 'Needs attention', className: 'state-stale' }
      : raw === 'offline' || raw === 'stopped'
        ? { label: 'Ready for review', className: 'state-offline' }
        : { label: 'Queued', className: 'state-not_started' };
  return (
    <article className="merge-card">
      <div className="merge-card-header">
        <div>
          <div className="merge-branch">{item.branch}</div>
          <div className="merge-track">
            <span>{item.agent}</span>
            <span className="merge-arrow">-&gt;</span>
            <span>{item.merge_target}</span>
          </div>
        </div>
        <span className={classNames('chip', status.className)}>{status.label}</span>
      </div>
      <div className="merge-meta">
        <div><strong>Submit</strong> {item.submit_strategy}</div>
        <div><strong>Worker identity</strong> {item.worker_identity}</div>
        <div><strong>Manager</strong> {item.manager_identity}</div>
      </div>
      <div className="merge-note">{item.manager_action}</div>
    </article>
  );
}

function AgentCard({ item }: { item: AgentRow }) {
  const processLine = item.process_alive ? `pid ${item.pid}` : item.last_seen || 'no heartbeat yet';
  const detailLine = item.process_alive ? 'process alive' : item.evidence || item.expected_next_checkin || 'waiting for launch';
  return (
    <article className="agent-card">
      <header>
        <div>
          <div className="agent-name">{item.agent}</div>
          <div className="agent-role">{item.role}</div>
        </div>
        <span className={classNames('chip', stateClass(item.display_state))}>{displayState(item.display_state)}</span>
      </header>
      <div className="agent-meta">
        <div><strong>Pool</strong> {item.resource_pool} / {item.provider}</div>
        <div><strong>Model</strong> {item.model}</div>
        <div><strong>Branch</strong> {item.branch}</div>
        <div><strong>Heartbeat</strong> {processLine}</div>
        <div className="muted">{detailLine}</div>
      </div>
    </article>
  );
}

function renderValidation(data: DashboardState): string {
  const launchBlockers = data.launch_blockers || [];
  const notes = data.validation_errors || [];
  const lines = [
    launchBlockers.length ? `Launch blockers:\n- ${launchBlockers.join('\n- ')}` : 'Launch blockers:\nnone',
    notes.length ? `\nConfig notes:\n- ${notes.join('\n- ')}` : '\nConfig notes:\nnone',
  ];
  return lines.join('\n');
}

async function writeClipboard(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}

export function App() {
  const [tab, setTab] = useState<TabKey>('overview');
  const [data, setData] = useState<DashboardState | null>(null);
  const [configText, setConfigText] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [actionInFlight, setActionInFlight] = useState(false);
  const [status, setStatus] = useState<{ message: string; error: boolean }>({ message: '', error: false });
  const abortRef = useRef<AbortController | null>(null);

  const agentRows = useMemo(() => buildAgentRows(data), [data]);
  const progress = useMemo(() => buildProgressModel(data, agentRows), [data, agentRows]);

  const setStampedStatus = (message: string, error = false) => {
    const stamp = new Date().toLocaleTimeString();
    setStatus({ message: `[${stamp}] ${message}`, error });
  };

  const refresh = async (forceStatus = false) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const nextData = await fetchState(controller.signal);
      setData(nextData);
      if (!editorDirty) {
        setConfigText(nextData.config_text || '');
      }
      if (forceStatus) {
        setStampedStatus(`state refreshed, last event: ${nextData.last_event || 'none'}`);
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        setStampedStatus(`refresh failed: ${String(error)}`, true);
      }
    }
  };

  useEffect(() => {
    void refresh(true);
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!autoRefresh || actionInFlight) {
      return;
    }
    const timer = window.setInterval(() => {
      void refresh(false);
    }, AUTO_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [autoRefresh, actionInFlight, editorDirty]);

  const runAction = async (label: string, action: () => Promise<void>) => {
    if (actionInFlight) {
      return;
    }
    setActionInFlight(true);
    setStampedStatus(`${label}...`);
    try {
      await action();
    } catch (error) {
      setStampedStatus(String(error), true);
    } finally {
      setActionInFlight(false);
    }
  };

  const onSave = () => void runAction('saving settings', async () => {
    const response = await saveConfig(configText);
    setEditorDirty(false);
    await refresh(true);
    setStampedStatus(`settings saved: ${response.launch_blockers.length} launch blocker(s), ${response.validation_errors.length} config note(s)`);
  });

  const onLaunch = (restart: boolean) => void runAction(restart ? 'restarting workers' : 'launching workers', async () => {
    const response = await launchWorkers(restart);
    setStampedStatus(`launch complete: ${(response.launched || []).length} launched, ${(response.failures || []).length} failures`, !response.ok);
    await refresh(true);
  });

  const onStopWorkers = () => void runAction('stopping workers', async () => {
    const response = await stopWorkers();
    setStampedStatus(`stopped workers: ${response.stopped.join(', ') || 'none'}`);
    await refresh(true);
  });

  const onStopAll = () => void runAction('stopping listener and workers', async () => {
    const response = await stopAll();
    setStampedStatus(
      response.listener_released
        ? `stop all requested: ${response.stopped_workers?.length || 0} worker(s) stopped, port ${response.listener_port} released`
        : `stop all requested${response.warning ? `: ${response.warning}` : ''}`,
    );
  });

  const onCopy = (mode: 'serve' | 'up') => void runAction(`copying ${mode} command`, async () => {
    if (!data?.commands[mode]) {
      throw new Error(`no ${mode} command available`);
    }
    await writeClipboard(data.commands[mode]);
    setStampedStatus(`${mode} command copied`);
  });

  const topMeta = data ? [
    { label: 'Startup', value: data.mode.state || 'configured' },
    { label: 'Launch', value: data.launch_blockers.length ? `${data.launch_blockers.length} blocker(s)` : 'ready' },
    { label: 'Config', value: data.mode.config_path || 'unknown' },
    { label: 'Last event', value: data.last_event || 'none' },
    { label: 'Updated', value: data.updated_at || 'unknown' },
  ] : [];

  return (
    <div>
      <header>
        <div className="hero">
          <div>
            <div className="hero-badge">FP8 delivery orchestration</div>
            <h1>supersonic-moe control plane</h1>
            <p className="small tagline">React-structured cold-start control for agent launch, inspection, settings, and deterministic stop behavior.</p>
          </div>
        </div>
      </header>
      <main>
        <section className="card">
          <div className="toolbar">
            <div className="toolbar-group">
              <button disabled={actionInFlight} onClick={() => onLaunch(false)}>Launch</button>
              <button className="secondary" disabled={actionInFlight} onClick={() => onLaunch(true)}>Restart</button>
              <button className="danger" disabled={actionInFlight} onClick={onStopWorkers}>Stop Agents</button>
              <button className="danger ghost-danger" disabled={actionInFlight} onClick={onStopAll}>Stop All</button>
              <button className="ghost" disabled={actionInFlight} onClick={() => void refresh(true)}>Refresh</button>
            </div>
            <div className="toolbar-group">
              <button className="ghost" disabled={actionInFlight} onClick={() => onCopy('serve')}>Copy Serve</button>
              <button className="ghost" disabled={actionInFlight} onClick={() => onCopy('up')}>Copy Up</button>
              <label className="toggle"><input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} /> Auto refresh</label>
            </div>
          </div>
          <div className={classNames('status', status.error && 'error')}>{status.message}</div>
        </section>

        <section className="card">
          <div className="toolbar">
            <div className="tab-nav" role="tablist" aria-label="Dashboard sections">
              {(['overview', 'operations', 'settings'] as TabKey[]).map((name) => (
                <button key={name} className={classNames('nav-button', tab === name && 'active')} onClick={() => setTab(name)}>{name[0].toUpperCase() + name.slice(1)}</button>
              ))}
            </div>
            <div className="pill-row">
              {topMeta.map((item) => <div key={item.label} className="key-pair"><span className="muted">{item.label}</span><strong>{item.value}</strong></div>)}
            </div>
          </div>
        </section>

        {data ? (
          tab === 'overview' ? <OverviewTab data={data} agentRows={agentRows} progress={progress} /> : tab === 'operations' ? <OperationsTab data={data} /> : <SettingsTab data={data} configText={configText} onChange={(value) => { setEditorDirty(true); setConfigText(value); }} onSave={onSave} />
        ) : (
          <section className="card"><div className="small muted">Loading dashboard state...</div></section>
        )}
      </main>
    </div>
  );
}
