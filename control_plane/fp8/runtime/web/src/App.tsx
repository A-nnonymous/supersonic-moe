import { useEffect, useMemo, useRef, useState } from 'react';
import { enableSilentMode, fetchState, launchWorkers, saveConfig, stopAll, stopWorkers, validateConfig } from './api';
import type {
  ConfigResourcePool,
  ConfigShape,
  ConfigWorker,
  DashboardState,
  GateItem,
  HeartbeatAgent,
  LaunchStrategy,
  MergeQueueItem,
  ProcessSnapshot,
  RuntimeWorker,
  TabKey,
  ValidationIssue,
} from './types';

const AUTO_REFRESH_MS = 4000;

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

type IssueMap = Record<string, string[]>;

function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ');
}

function displayState(value: string | undefined): string {
  return String(value || 'unknown').replaceAll('_', ' ');
}

function stateClass(value: string | undefined): string {
  return `state-${String(value || 'unknown').replace(/[^a-zA-Z0-9]+/g, '_')}`;
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

function cloneConfig(config: ConfigShape | undefined): ConfigShape {
  if (!config) {
    return { project: {}, providers: {}, resource_pools: {}, workers: [] };
  }
  return JSON.parse(JSON.stringify(config)) as ConfigShape;
}

function normalizeConfig(config: ConfigShape): ConfigShape {
  return {
    project: config.project || {},
    providers: config.providers || {},
    resource_pools: config.resource_pools || {},
    workers: config.workers || [],
  };
}

function buildIssueMap(...issueSets: ValidationIssue[][]): IssueMap {
  return issueSets.reduce<IssueMap>((acc, issues) => {
    issues.forEach((issue) => {
      acc[issue.field] = [...(acc[issue.field] || []), issue.message];
    });
    return acc;
  }, {});
}

function parseQueue(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function stringifyQueue(values: string[] | undefined): string {
  return (values || []).join(', ');
}

function launchStrategyLabel(strategy: LaunchStrategy): string {
  if (strategy === 'initial_copilot') {
    return 'Initial Copilot';
  }
  if (strategy === 'selected_model') {
    return 'Selected Model';
  }
  return 'Elastic';
}

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
    remember(item.agent, { branch: item.branch });
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

function getLocalValidationIssues(config: ConfigShape): ValidationIssue[] {
  const draft = normalizeConfig(config);
  const issues: ValidationIssue[] = [];
  const add = (field: string, message: string) => issues.push({ field, message });
  const project = draft.project || {};
  const dashboard = project.dashboard || {};
  const pools = draft.resource_pools || {};
  const workers = draft.workers || [];

  if (!String(project.repository_name || '').trim()) {
    add('project.repository_name', 'repository name is required');
  }
  if (!String(project.local_repo_root || '').trim()) {
    add('project.local_repo_root', 'local repo root is required');
  }
  if (!String(project.paddle_repo_path || '').trim()) {
    add('project.paddle_repo_path', 'Paddle path is required');
  }
  if (!String(dashboard.host || '').trim()) {
    add('project.dashboard.host', 'dashboard host is required');
  }
  if (!Number.isInteger(Number(dashboard.port)) || Number(dashboard.port) < 1 || Number(dashboard.port) > 65535) {
    add('project.dashboard.port', 'dashboard port must be between 1 and 65535');
  }
  if (!String(project.integration_branch || project.base_branch || '').trim()) {
    add('project.integration_branch', 'integration branch is required');
  }

  const seenAgents = new Set<string>();
  const seenBranches = new Set<string>();
  const seenWorktrees = new Set<string>();

  Object.entries(pools).forEach(([poolName, pool]) => {
    if (!String(pool.provider || '').trim()) {
      add(`resource_pools.${poolName}.provider`, 'provider is required');
    }
    if (!String(pool.model || '').trim()) {
      add(`resource_pools.${poolName}.model`, 'model is required');
    }
    if (!Number.isInteger(Number(pool.priority ?? 100))) {
      add(`resource_pools.${poolName}.priority`, 'priority must be an integer');
    }
  });

  workers.forEach((worker, index) => {
    const root = `workers[${index}]`;
    const agent = String(worker.agent || '').trim();
    const branch = String(worker.branch || '').trim();
    const worktreePath = String(worker.worktree_path || '').trim();
    if (!agent) {
      add(`${root}.agent`, 'agent is required');
    } else if (seenAgents.has(agent)) {
      add(`${root}.agent`, 'agent must be unique');
    } else {
      seenAgents.add(agent);
    }
    if (!branch) {
      add(`${root}.branch`, 'branch is required');
    } else if (seenBranches.has(branch)) {
      add(`${root}.branch`, 'branch must be unique');
    } else {
      seenBranches.add(branch);
    }
    if (!worktreePath) {
      add(`${root}.worktree_path`, 'worktree path is required');
    } else if (seenWorktrees.has(worktreePath)) {
      add(`${root}.worktree_path`, 'worktree path must be unique');
    } else {
      seenWorktrees.add(worktreePath);
    }
    const poolName = String(worker.resource_pool || '').trim();
    const queue = worker.resource_pool_queue || [];
    if (!poolName && !queue.length) {
      add(`${root}.resource_pool`, 'resource pool or queue is required');
    }
    if (!String(worker.test_command || '').trim()) {
      add(`${root}.test_command`, 'test command is required');
    }
    if (!String(worker.submit_strategy || '').trim()) {
      add(`${root}.submit_strategy`, 'submit strategy is required');
    }
    if (String(worker.environment_type || 'uv') !== 'none' && !String(worker.environment_path || '').trim()) {
      add(`${root}.environment_path`, 'environment path is required unless environment type is none');
    }
  });

  return issues;
}

function DataTable({ columns, rows }: { columns: string[]; rows: Array<Record<string, unknown>> }) {
  if (!rows.length) {
    return <div className="small muted">No data</div>;
  }
  return (
    <div className="table-shell">
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
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  issues,
  placeholder,
  type = 'text',
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  issues?: string[];
  placeholder?: string;
  type?: 'text' | 'number';
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input
        className={classNames('field-input', issues && issues.length > 0 && 'field-input-error')}
        type={type}
        value={value ?? ''}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
      {issues && issues.length > 0 ? <span className="field-error">{issues[0]}</span> : null}
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  issues,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  issues?: string[];
  options: string[];
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select className={classNames('field-input', issues && issues.length > 0 && 'field-input-error')} value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">Select…</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      {issues && issues.length > 0 ? <span className="field-error">{issues[0]}</span> : null}
    </label>
  );
}

function SectionIssueList({ issues }: { issues: ValidationIssue[] }) {
  if (!issues.length) {
    return null;
  }
  return (
    <div className="settings-issues">
      <h3>Validation Warnings</h3>
      <ul>
        {issues.map((issue, index) => <li key={`${issue.field}-${index}`}>{issue.field}: {issue.message}</li>)}
      </ul>
    </div>
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
    { key: 'listener_active', value: data.mode.listener_active },
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

function SettingsTab({
  draftConfig,
  providerOptions,
  issues,
  backendIssues,
  onProjectChange,
  onMergeChange,
  onPoolChange,
  onAddPool,
  onWorkerChange,
  onAddWorker,
  onSave,
}: {
  draftConfig: ConfigShape;
  providerOptions: string[];
  issues: IssueMap;
  backendIssues: ValidationIssue[];
  onProjectChange: (field: string, value: string) => void;
  onMergeChange: (field: string, value: string) => void;
  onPoolChange: (poolName: string, field: keyof ConfigResourcePool, value: string) => void;
  onAddPool: () => void;
  onWorkerChange: (index: number, field: string, value: string) => void;
  onAddWorker: () => void;
  onSave: () => void;
}) {
  const project = draftConfig.project || {};
  const dashboard = project.dashboard || {};
  const pools = draftConfig.resource_pools || {};
  const workers = draftConfig.workers || [];
  return (
    <div className="tab-body">
      <section className="card">
        <div className="page-header">
          <div>
            <h2>Settings</h2>
            <p className="small">The four cards below are editable. Invalid values are warned and are not saved.</p>
          </div>
          <button onClick={onSave}>Validate And Save</button>
        </div>
        <SectionIssueList issues={backendIssues} />
        <div className="settings-grid">
          <section className="helper-card settings-card">
            <div className="section-head"><h3>Project</h3></div>
            <div className="field-grid">
              <Field label="Repository" value={project.repository_name || ''} onChange={(value) => onProjectChange('repository_name', value)} issues={issues['project.repository_name']} />
              <Field label="Local Repo Root" value={project.local_repo_root || ''} onChange={(value) => onProjectChange('local_repo_root', value)} issues={issues['project.local_repo_root']} />
              <Field label="Paddle Repo Path" value={project.paddle_repo_path || ''} onChange={(value) => onProjectChange('paddle_repo_path', value)} issues={issues['project.paddle_repo_path']} />
              <Field label="Dashboard Host" value={dashboard.host || ''} onChange={(value) => onProjectChange('dashboard.host', value)} issues={issues['project.dashboard.host']} />
              <Field label="Dashboard Port" type="number" value={dashboard.port || 8233} onChange={(value) => onProjectChange('dashboard.port', value)} issues={issues['project.dashboard.port']} />
            </div>
          </section>

          <section className="helper-card settings-card">
            <div className="section-head">
              <h3>Resource Pools</h3>
              <button className="ghost" type="button" onClick={onAddPool}>Add Pool</button>
            </div>
            <div className="stack-list">
              {Object.entries(pools).map(([poolName, pool]) => (
                <div key={poolName} className="subcard">
                  <div className="subcard-title">{poolName}</div>
                  <div className="field-grid">
                    <SelectField label="Provider" value={String(pool.provider || '')} onChange={(value) => onPoolChange(poolName, 'provider', value)} issues={issues[`resource_pools.${poolName}.provider`]} options={providerOptions} />
                    <Field label="Model" value={String(pool.model || '')} onChange={(value) => onPoolChange(poolName, 'model', value)} issues={issues[`resource_pools.${poolName}.model`]} />
                    <Field label="Priority" type="number" value={Number(pool.priority ?? 100)} onChange={(value) => onPoolChange(poolName, 'priority', value)} issues={issues[`resource_pools.${poolName}.priority`]} />
                    <Field label="API Key" value={String(pool.api_key || '')} onChange={(value) => onPoolChange(poolName, 'api_key', value)} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="helper-card settings-card">
            <div className="section-head"><h3>Merge Policy</h3></div>
            <div className="field-grid">
              <Field label="Integration Branch" value={project.integration_branch || project.base_branch || ''} onChange={(value) => onMergeChange('integration_branch', value)} issues={issues['project.integration_branch']} />
              <Field label="Manager Name" value={project.manager_git_identity?.name || ''} onChange={(value) => onMergeChange('manager_git_identity.name', value)} />
              <Field label="Manager Email" value={project.manager_git_identity?.email || ''} onChange={(value) => onMergeChange('manager_git_identity.email', value)} />
            </div>
          </section>

          <section className="helper-card settings-card settings-card-wide">
            <div className="section-head">
              <h3>Worker Config</h3>
              <button className="ghost" type="button" onClick={onAddWorker}>Add Worker</button>
            </div>
            <div className="stack-list">
              {workers.map((worker, index) => (
                <div key={`${worker.agent || 'worker'}-${index}`} className="subcard">
                  <div className="subcard-title">{worker.agent || `Worker ${index + 1}`}</div>
                  <div className="field-grid">
                    <Field label="Agent" value={worker.agent || ''} onChange={(value) => onWorkerChange(index, 'agent', value)} issues={issues[`workers[${index}].agent`]} />
                    <Field label="Task ID" value={worker.task_id || ''} onChange={(value) => onWorkerChange(index, 'task_id', value)} />
                    <Field label="Resource Pool" value={worker.resource_pool || ''} onChange={(value) => onWorkerChange(index, 'resource_pool', value)} issues={issues[`workers[${index}].resource_pool`]} />
                    <Field label="Pool Queue" value={stringifyQueue(worker.resource_pool_queue)} onChange={(value) => onWorkerChange(index, 'resource_pool_queue', value)} issues={issues[`workers[${index}].resource_pool_queue`]} placeholder="pool_a, pool_b" />
                    <Field label="Branch" value={worker.branch || ''} onChange={(value) => onWorkerChange(index, 'branch', value)} issues={issues[`workers[${index}].branch`]} />
                    <Field label="Worktree Path" value={worker.worktree_path || ''} onChange={(value) => onWorkerChange(index, 'worktree_path', value)} issues={issues[`workers[${index}].worktree_path`]} />
                    <SelectField label="Environment Type" value={worker.environment_type || 'uv'} onChange={(value) => onWorkerChange(index, 'environment_type', value)} options={['uv', 'venv', 'none']} />
                    <Field label="Environment Path" value={worker.environment_path || ''} onChange={(value) => onWorkerChange(index, 'environment_path', value)} issues={issues[`workers[${index}].environment_path`]} />
                    <Field label="Sync Command" value={worker.sync_command || ''} onChange={(value) => onWorkerChange(index, 'sync_command', value)} />
                    <Field label="Test Command" value={worker.test_command || ''} onChange={(value) => onWorkerChange(index, 'test_command', value)} issues={issues[`workers[${index}].test_command`]} />
                    <Field label="Submit Strategy" value={worker.submit_strategy || ''} onChange={(value) => onWorkerChange(index, 'submit_strategy', value)} issues={issues[`workers[${index}].submit_strategy`]} />
                    <Field label="Git Name" value={worker.git_identity?.name || ''} onChange={(value) => onWorkerChange(index, 'git_identity.name', value)} />
                    <Field label="Git Email" value={worker.git_identity?.email || ''} onChange={(value) => onWorkerChange(index, 'git_identity.email', value)} />
                  </div>
                </div>
              ))}
            </div>
          </section>
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
  const [draftConfig, setDraftConfig] = useState<ConfigShape>({ project: {}, providers: {}, resource_pools: {}, workers: [] });
  const [configDirty, setConfigDirty] = useState(false);
  const [launchStrategy, setLaunchStrategy] = useState<LaunchStrategy>('initial_copilot');
  const [launchProvider, setLaunchProvider] = useState('copilot');
  const [launchModel, setLaunchModel] = useState('');
  const [launchDirty, setLaunchDirty] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [actionInFlight, setActionInFlight] = useState(false);
  const [status, setStatus] = useState<{ message: string; error: boolean }>({ message: '', error: false });
  const [backendIssues, setBackendIssues] = useState<ValidationIssue[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const agentRows = useMemo(() => buildAgentRows(data), [data]);
  const progress = useMemo(() => buildProgressModel(data, agentRows), [data, agentRows]);
  const localIssues = useMemo(() => getLocalValidationIssues(draftConfig), [draftConfig]);
  const issueMap = useMemo(() => buildIssueMap(localIssues, backendIssues), [localIssues, backendIssues]);
  const providerOptions = useMemo(() => Object.keys(draftConfig.providers || {}), [draftConfig.providers]);

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
      if (!configDirty) {
        setDraftConfig(cloneConfig(nextData.config));
        setBackendIssues([]);
      }
      if (!launchDirty) {
        setLaunchStrategy(nextData.launch_policy.default_strategy);
        setLaunchProvider(nextData.launch_policy.default_provider || nextData.launch_policy.initial_provider || 'copilot');
        setLaunchModel(nextData.launch_policy.default_model || '');
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
  }, [autoRefresh, actionInFlight, configDirty, launchDirty]);

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

  const updateConfig = (updater: (current: ConfigShape) => ConfigShape) => {
    setConfigDirty(true);
    setBackendIssues([]);
    setDraftConfig((current) => normalizeConfig(updater(normalizeConfig(cloneConfig(current)))));
  };

  const onProjectChange = (field: string, value: string) => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      next.project = next.project || {};
      if (field.startsWith('dashboard.')) {
        const key = field.replace('dashboard.', '');
        next.project.dashboard = next.project.dashboard || {};
        if (key === 'port') {
          next.project.dashboard.port = Number(value);
        } else {
          next.project.dashboard.host = value;
        }
      } else if (field === 'repository_name') {
        next.project.repository_name = value;
      } else if (field === 'local_repo_root') {
        next.project.local_repo_root = value;
      } else if (field === 'paddle_repo_path') {
        next.project.paddle_repo_path = value;
      }
      return next;
    });
  };

  const onMergeChange = (field: string, value: string) => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      next.project = next.project || {};
      if (field === 'integration_branch') {
        next.project.integration_branch = value;
      } else {
        next.project.manager_git_identity = next.project.manager_git_identity || {};
        if (field.endsWith('.name')) {
          next.project.manager_git_identity.name = value;
        } else {
          next.project.manager_git_identity.email = value;
        }
      }
      return next;
    });
  };

  const onPoolChange = (poolName: string, field: keyof ConfigResourcePool, value: string) => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      next.resource_pools = next.resource_pools || {};
      const existing = next.resource_pools[poolName] || {};
      next.resource_pools[poolName] = {
        ...existing,
        [field]: field === 'priority' ? Number(value) : value,
      };
      return next;
    });
  };

  const onAddPool = () => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      next.resource_pools = next.resource_pools || {};
      let index = Object.keys(next.resource_pools).length + 1;
      let name = `pool_${index}`;
      while (next.resource_pools[name]) {
        index += 1;
        name = `pool_${index}`;
      }
      next.resource_pools[name] = { priority: 100, provider: providerOptions[0] || '', model: '', api_key: '' };
      return next;
    });
  };

  const onWorkerChange = (index: number, field: string, value: string) => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const workers = [...(next.workers || [])];
      const worker = { ...(workers[index] || { agent: `A${index + 1}` }) };
      if (field === 'resource_pool_queue') {
        worker.resource_pool_queue = parseQueue(value);
      } else if (field === 'git_identity.name' || field === 'git_identity.email') {
        worker.git_identity = worker.git_identity || {};
        if (field.endsWith('.name')) {
          worker.git_identity.name = value;
        } else {
          worker.git_identity.email = value;
        }
      } else {
        (worker as Record<string, unknown>)[field] = value;
      }
      workers[index] = worker;
      next.workers = workers;
      return next;
    });
  };

  const onAddWorker = () => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const workers = [...(next.workers || [])];
      workers.push({
        agent: `A${workers.length + 1}`,
        task_id: '',
        resource_pool: Object.keys(next.resource_pools || {})[0] || '',
        resource_pool_queue: [],
        worktree_path: '',
        branch: '',
        environment_type: 'uv',
        environment_path: '',
        sync_command: 'uv sync',
        test_command: '',
        submit_strategy: 'patch_handoff',
        git_identity: { name: '', email: '' },
      });
      next.workers = workers;
      return next;
    });
  };

  const onSave = () => void runAction('validating settings', async () => {
    if (localIssues.length > 0) {
      setStampedStatus(`settings contain ${localIssues.length} local validation issue(s)`, true);
      return;
    }
    const validation = await validateConfig(draftConfig);
    setBackendIssues(validation.validation_issues);
    if (!validation.ok) {
      setStampedStatus(`settings rejected: ${validation.validation_issues.length} validation issue(s)`, true);
      return;
    }
    const response = await saveConfig(draftConfig);
    setConfigDirty(false);
    setBackendIssues([]);
    await refresh(true);
    setStampedStatus(`settings saved: ${response.launch_blockers.length} launch blocker(s), ${response.validation_errors.length} config note(s)`);
  });

  const onLaunch = (restart: boolean) => void runAction(restart ? 'restarting workers' : 'launching workers', async () => {
    const response = await launchWorkers(restart, {
      strategy: launchStrategy,
      provider: launchStrategy === 'elastic' ? undefined : launchProvider,
      model: launchStrategy === 'selected_model' ? launchModel : undefined,
    });
    setStampedStatus(
      `launch complete (${launchStrategyLabel(response.launch_policy?.strategy || launchStrategy)}): ${(response.launched || []).length} launched, ${(response.failures || []).length} failures`,
      !response.ok,
    );
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
      !response.listener_released,
    );
  });

  const onSilentMode = () => void runAction('entering silent mode', async () => {
    const response = await enableSilentMode();
    setStampedStatus(`silent mode enabled: listener on port ${response.listener_port} closed`);
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
    { label: 'Listener', value: data.mode.listener_active ? 'active' : 'silent' },
    { label: 'Launch', value: data.launch_blockers.length ? `${data.launch_blockers.length} blocker(s)` : 'ready' },
    { label: 'Launch Mode', value: launchStrategyLabel(launchStrategy) },
    { label: 'Config', value: data.mode.config_path || 'unknown' },
    { label: 'Updated', value: data.updated_at || 'unknown' },
  ] : [];

  return (
    <div>
      <header>
        <div className="hero">
          <div>
            <div className="hero-badge">FP8 delivery orchestration</div>
            <h1>supersonic-moe control plane</h1>
            <p className="small tagline">Cold-start by default, fire-and-forget serving, editable settings forms, strict validation, and an explicit silent listener mode.</p>
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
              <button className="ghost danger-outline" disabled={actionInFlight} onClick={onSilentMode}>Silent Mode</button>
              <button className="danger ghost-danger" disabled={actionInFlight} onClick={onStopAll}>Stop All</button>
              <button className="ghost" disabled={actionInFlight} onClick={() => void refresh(true)}>Refresh</button>
            </div>
            <div className="toolbar-group">
              {data ? (
                <>
                  <label className="field field-compact">
                    <span className="field-label">Launch Mode</span>
                    <select
                      className="field-input compact-input"
                      value={launchStrategy}
                      onChange={(event) => {
                        setLaunchDirty(true);
                        setLaunchStrategy(event.target.value as LaunchStrategy);
                        if (event.target.value === 'initial_copilot') {
                          setLaunchProvider(data.launch_policy.initial_provider || 'copilot');
                        }
                      }}
                    >
                      {data.launch_policy.available_strategies.map((strategy) => (
                        <option key={strategy} value={strategy}>{launchStrategyLabel(strategy)}</option>
                      ))}
                    </select>
                  </label>
                  {launchStrategy !== 'elastic' ? (
                    <label className="field field-compact">
                      <span className="field-label">Provider</span>
                      <select
                        className="field-input compact-input"
                        value={launchProvider}
                        disabled={launchStrategy === 'initial_copilot'}
                        onChange={(event) => {
                          setLaunchDirty(true);
                          setLaunchProvider(event.target.value);
                        }}
                      >
                        {data.launch_policy.available_providers.map((provider) => (
                          <option key={provider} value={provider}>{provider}</option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                  {launchStrategy === 'selected_model' ? (
                    <label className="field field-compact field-compact-wide">
                      <span className="field-label">Model</span>
                      <input
                        className="field-input compact-input"
                        value={launchModel}
                        placeholder="gpt-5.4"
                        onChange={(event) => {
                          setLaunchDirty(true);
                          setLaunchModel(event.target.value);
                        }}
                      />
                    </label>
                  ) : null}
                </>
              ) : null}
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
          tab === 'overview'
            ? <OverviewTab data={data} agentRows={agentRows} progress={progress} />
            : tab === 'operations'
              ? <OperationsTab data={data} />
              : <SettingsTab
                  draftConfig={draftConfig}
                  providerOptions={providerOptions}
                  issues={issueMap}
                  backendIssues={backendIssues}
                  onProjectChange={onProjectChange}
                  onMergeChange={onMergeChange}
                  onPoolChange={onPoolChange}
                  onAddPool={onAddPool}
                  onWorkerChange={onWorkerChange}
                  onAddWorker={onAddWorker}
                  onSave={onSave}
                />
        ) : (
          <section className="card"><div className="small muted">Loading dashboard state...</div></section>
        )}
      </main>
    </div>
  );
}
