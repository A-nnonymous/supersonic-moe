import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { enableSilentMode, fetchState, launchWorkers, saveConfig, saveConfigSection, stopAll, stopWorkers, validateConfig, validateConfigSection } from './api';
import type {
  ConfigSection,
  ConfigResourcePool,
  ConfigShape,
  ConfigWorker,
  ConfigWorkerDefaults,
  DashboardState,
  GateItem,
  HeartbeatAgent,
  LaunchStrategy,
  MergeQueueItem,
  ProcessSnapshot,
  ResolvedWorkerPlan,
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
type SectionStatusMap = Partial<Record<ConfigSection, { message: string; error: boolean }>>;
type PlannedWorker = {
  agent: string;
  task_id: string;
  title: string;
  branch: string;
  worktree_path: string;
};

type WorkerPlanView = {
  agent: string;
  taskId: string;
  taskTitle: string;
  taskType: string;
  branch: string;
  worktreePath: string;
  poolOverride: string;
  queueOverride: string[];
  recommendedPool: string;
  lockedPool: string;
  poolReason: string;
  testCommand: string;
  suggestedTestCommand: string;
};

type WorkerResetScope = 'all' | 'routing' | 'runtime';

function normalizedText(value: unknown): string {
  return String(value ?? '').trim();
}

function isAutoManagedBlank(value: unknown): boolean {
  const normalized = normalizedText(value).toLowerCase();
  return !normalized || normalized === 'unassigned';
}

function isAutoManagedCommandBlank(value: unknown): boolean {
  const normalized = normalizedText(value).toLowerCase();
  return !normalized || normalized === 'unassigned' || normalized === 'none';
}

function isPlaceholderPath(value: unknown): boolean {
  const normalized = normalizedText(value);
  return Boolean(normalized) && (normalized.startsWith('/absolute/path/') || normalized === 'unassigned' || normalized === 'none');
}

function firstMeaningfulValue(...values: unknown[]): string {
  for (const value of values) {
    if (!isAutoManagedBlank(value)) {
      return normalizedText(value);
    }
  }
  return '';
}

function firstMeaningfulCommand(...values: unknown[]): string {
  for (const value of values) {
    if (!isAutoManagedCommandBlank(value)) {
      return normalizedText(value);
    }
  }
  return '';
}

function firstMeaningfulPath(...values: unknown[]): string {
  for (const value of values) {
    if (!isAutoManagedBlank(value) && !isPlaceholderPath(value)) {
      return normalizedText(value);
    }
  }
  return '';
}

function projectReferenceWorkspace(project: ConfigShape['project']): string {
  return normalizedText(project?.reference_workspace_root) || normalizedText(project?.paddle_repo_path);
}

function mergeSavedSection(current: ConfigShape, server: ConfigShape, section: ConfigSection): ConfigShape {
  const next = normalizeConfig(cloneConfig(current));
  const serverConfig = normalizeConfig(cloneConfig(server));
  if (section === 'project') {
    next.project = {
      ...(next.project || {}),
      repository_name: serverConfig.project?.repository_name,
      local_repo_root: serverConfig.project?.local_repo_root,
      reference_workspace_root: serverConfig.project?.reference_workspace_root,
      paddle_repo_path: serverConfig.project?.paddle_repo_path,
      dashboard: serverConfig.project?.dashboard || {},
    };
    return next;
  }
  if (section === 'merge_policy') {
    next.project = {
      ...(next.project || {}),
      integration_branch: serverConfig.project?.integration_branch,
      base_branch: serverConfig.project?.base_branch,
      manager_git_identity: serverConfig.project?.manager_git_identity || {},
    };
    return next;
  }
  if (section === 'resource_pools') {
    next.resource_pools = serverConfig.resource_pools || {};
    return next;
  }
  if (section === 'worker_defaults') {
    next.worker_defaults = serverConfig.worker_defaults || {};
    return next;
  }
  next.workers = serverConfig.workers || [];
  return next;
}

function buildSectionValue(config: ConfigShape, section: ConfigSection): unknown {
  if (section === 'project') {
    const project = config.project || {};
    return {
      repository_name: project.repository_name || '',
      local_repo_root: project.local_repo_root || '',
      reference_workspace_root: projectReferenceWorkspace(project),
      dashboard: project.dashboard || {},
    };
  }
  if (section === 'merge_policy') {
    const project = config.project || {};
    return {
      integration_branch: project.integration_branch || project.base_branch || '',
      manager_git_identity: project.manager_git_identity || {},
    };
  }
  if (section === 'resource_pools') {
    return config.resource_pools || {};
  }
  if (section === 'worker_defaults') {
    return config.worker_defaults || {};
  }
  return config.workers || [];
}

function configForProjectSave(baseConfig: ConfigShape | undefined, draftConfig: ConfigShape): ConfigShape {
  const base = normalizeConfig(cloneConfig(baseConfig));
  const draft = normalizeConfig(cloneConfig(draftConfig));
  return normalizeDerivedPaths(mergeSavedSection(base, draft, 'project'));
}

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
    return { project: {}, providers: {}, resource_pools: {}, worker_defaults: {}, workers: [] };
  }
  return JSON.parse(JSON.stringify(config)) as ConfigShape;
}

function normalizeConfig(config: ConfigShape): ConfigShape {
  return {
    project: config.project || {},
    providers: config.providers || {},
    resource_pools: config.resource_pools || {},
    worker_defaults: config.worker_defaults || {},
    workers: config.workers || [],
  };
}

function mergeWorkerWithDefaults(worker: ConfigWorker, defaults: ConfigWorkerDefaults | undefined): ConfigWorker {
  const merged: ConfigWorker = { ...worker };
  const workerDefaults = defaults || {};
  const inheritableFields: Array<keyof ConfigWorkerDefaults> = [
    'resource_pool',
    'environment_type',
    'environment_path',
    'sync_command',
    'submit_strategy',
    'test_command',
  ];

  inheritableFields.forEach((field) => {
    const workerValue = merged[field as keyof ConfigWorker];
    const defaultValue = workerDefaults[field];
    if ((workerValue === undefined || workerValue === '') && defaultValue !== undefined && defaultValue !== '') {
      (merged as Record<string, unknown>)[field] = defaultValue;
    }
  });

  if ((!Array.isArray(merged.resource_pool_queue) || merged.resource_pool_queue.length === 0) && Array.isArray(workerDefaults.resource_pool_queue) && workerDefaults.resource_pool_queue.length > 0) {
    merged.resource_pool_queue = [...workerDefaults.resource_pool_queue];
  }

  const defaultIdentity = workerDefaults.git_identity || {};
  const workerIdentity = merged.git_identity || {};
  const mergedIdentity = {
    name: workerIdentity.name || defaultIdentity.name || '',
    email: workerIdentity.email || defaultIdentity.email || '',
  };
  if (mergedIdentity.name || mergedIdentity.email) {
    merged.git_identity = mergedIdentity;
  }

  return merged;
}

function resetWorkerDefaultsToA0(defaults: ConfigWorkerDefaults | undefined): ConfigWorkerDefaults {
  return {};
}

function resetWorkerOverridesToA0(worker: ConfigWorker, scope: WorkerResetScope = 'all'): ConfigWorker {
  const next: ConfigWorker = { agent: worker.agent };
  if (scope === 'routing') {
    return {
      ...worker,
      resource_pool: undefined,
      resource_pool_queue: undefined,
      task_id: undefined,
      branch: undefined,
      worktree_path: undefined,
    };
  }
  if (scope === 'runtime') {
    return {
      ...worker,
      environment_type: undefined,
      environment_path: undefined,
      sync_command: undefined,
      test_command: undefined,
      submit_strategy: undefined,
      git_identity: undefined,
    };
  }
  return next;
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

function preferredLaunchProvider(launchPolicy: DashboardState['launch_policy']): string {
  return launchPolicy.default_provider || launchPolicy.initial_provider || launchPolicy.available_providers[0] || '';
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .replace(/_{2,}/g, '_');
}

function deriveWorktreePath(config: ConfigShape, agent: string): string {
  const project = config.project || {};
  const localRepoRoot = String(project.local_repo_root || '').trim();
  if (!localRepoRoot) {
    return '';
  }
  const normalizedRoot = localRepoRoot.replace(/\/$/, '');
  const parent = normalizedRoot.includes('/') ? normalizedRoot.slice(0, normalizedRoot.lastIndexOf('/')) : normalizedRoot;
  const baseName = String(project.repository_name || normalizedRoot.split('/').pop() || 'supersonic-moe');
  return `${parent}/${slugify(baseName)}_${agent.toLowerCase()}`;
}

function deriveBranchName(agent: string, title: string, taskId: string): string {
  const branchSuffix = slugify(title || taskId || agent) || agent.toLowerCase();
  return `${agent.toLowerCase()}_${branchSuffix}`;
}

function deriveDefaultEnvironmentPath(config: ConfigShape): string {
  const localRepoRoot = normalizedText(config.project?.local_repo_root);
  if (!localRepoRoot) {
    return '';
  }
  return `${localRepoRoot.replace(/\/$/, '')}/.venv`;
}

function normalizeDerivedPaths(config: ConfigShape): ConfigShape {
  const next = normalizeConfig(cloneConfig(config));
  const nextAutoEnvPath = deriveDefaultEnvironmentPath(next);
  next.worker_defaults = next.worker_defaults || {};
  if (next.worker_defaults.environment_type !== 'none') {
    const currentEnvironmentPath = normalizedText(next.worker_defaults.environment_path);
    if (!currentEnvironmentPath || isPlaceholderPath(currentEnvironmentPath)) {
      next.worker_defaults.environment_path = nextAutoEnvPath;
    }
  }
  next.workers = (next.workers || []).map((worker) => {
    const worktreePath = normalizedText(worker.worktree_path);
    const environmentPath = normalizedText(worker.environment_path);
    return {
      ...worker,
      worktree_path: !worktreePath || isPlaceholderPath(worktreePath) ? deriveWorktreePath(next, worker.agent) : worker.worktree_path,
      environment_path: !environmentPath || isPlaceholderPath(environmentPath) ? undefined : worker.environment_path,
    };
  });
  return next;
}

function formatLaunchErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const lines = message.split('\n').map((line) => line.trim()).filter(Boolean);
  const pathBlockers = new Map<string, string[]>();
  for (const line of lines) {
    const match = line.match(/^worker\s+(A\d+)\s+(worktree_path|environment_path)\s+must be replaced with a real path$/);
    if (!match) {
      continue;
    }
    const [, agent, field] = match;
    const entry = pathBlockers.get(agent) || [];
    entry.push(field === 'worktree_path' ? 'worktree' : 'environment');
    pathBlockers.set(agent, entry);
  }
  if (!pathBlockers.size) {
    return message;
  }
  const workerSummary = Array.from(pathBlockers.entries())
    .map(([agent, fields]) => `${agent}: ${fields.join(' + ')} path`)
    .join('; ');
  return `Launch blocked: saved worker paths still contain template placeholders. ${workerSummary}. Save Project again to persist derived paths, or use Reset to A0 in Worker Config if a worker was manually pinned to a placeholder.`;
}

function deriveDefaultPoolQueue(config: ConfigShape, data: DashboardState | null): string[] {
  if (data?.provider_queue?.length) {
    return data.provider_queue.map((item) => item.resource_pool);
  }
  return Object.entries(config.resource_pools || {})
    .sort((left, right) => Number(right[1].priority ?? 100) - Number(left[1].priority ?? 100))
    .map(([poolName]) => poolName);
}

function inferRuntimeWorkerValue(data: DashboardState | null, field: keyof RuntimeWorker, allowNone = false): string {
  const counts = new Map<string, number>();
  (data?.runtime.workers || []).forEach((worker) => {
    if (worker.agent === 'A0') {
      return;
    }
    const value = normalizedText((worker as Record<string, unknown>)[field as string]);
    const lowered = value.toLowerCase();
    if (!value || lowered === 'unassigned' || (!allowNone && lowered === 'none')) {
      return;
    }
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  return Array.from(counts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] || '';
}

function buildRuntimeWorkerMap(data: DashboardState | null): Map<string, Record<string, unknown>> {
  return new Map((data?.runtime.workers || []).map((worker) => [worker.agent, worker as Record<string, unknown>]));
}

function buildResolvedWorkerMap(data: DashboardState | null): Map<string, ResolvedWorkerPlan> {
  return new Map((data?.resolved_workers || []).map((worker) => [worker.agent, worker]));
}

function workerPlanView(
  worker: ConfigWorker,
  config: ConfigShape,
  data: DashboardState | null,
  plannedWorkers?: PlannedWorker[],
): WorkerPlanView {
  const plannedByAgent = new Map((plannedWorkers || buildPlannedWorkers(data, config)).map((item) => [item.agent, item]));
  const resolvedByAgent = buildResolvedWorkerMap(data);
  const planned = plannedByAgent.get(worker.agent || '');
  const resolved = resolvedByAgent.get(worker.agent || '');
  return {
    agent: worker.agent || '',
    taskId: firstMeaningfulValue(worker.task_id, resolved?.task_id, planned?.task_id),
    taskTitle: firstMeaningfulValue(resolved?.task_title, planned?.title),
    taskType: firstMeaningfulValue(resolved?.task_type, resolved?.task_category),
    branch: firstMeaningfulValue(worker.branch, resolved?.branch, planned?.branch),
    worktreePath: firstMeaningfulValue(worker.worktree_path, resolved?.worktree_path, planned?.worktree_path),
    poolOverride: normalizedText(worker.resource_pool),
    queueOverride: Array.isArray(worker.resource_pool_queue) ? worker.resource_pool_queue : [],
    recommendedPool: normalizedText(resolved?.recommended_pool),
    lockedPool: normalizedText(resolved?.locked_pool),
    poolReason: normalizedText(resolved?.pool_reason),
    testCommand: firstMeaningfulCommand(worker.test_command, resolved?.test_command),
    suggestedTestCommand: firstMeaningfulCommand(resolved?.suggested_test_command),
  };
}

function hydrateConfigForA0(data: DashboardState | null, config: ConfigShape): ConfigShape {
  const next = normalizeConfig(cloneConfig(config));
  const runtimeByAgent = buildRuntimeWorkerMap(data);
  const resolvedByAgent = buildResolvedWorkerMap(data);
  const managerRuntime = runtimeByAgent.get('A0');

  next.project = next.project || {};
  next.project.repository_name = firstMeaningfulValue(next.project.repository_name, managerRuntime?.repository_name);
  next.project.local_repo_root = firstMeaningfulValue(
    next.project.local_repo_root,
    managerRuntime?.local_workspace_root,
    managerRuntime?.repository_root,
  );
  next.project.integration_branch = firstMeaningfulValue(next.project.integration_branch, next.project.base_branch, 'main');
  next.project.dashboard = next.project.dashboard || {};
  next.project.dashboard.host = firstMeaningfulValue(next.project.dashboard.host, '0.0.0.0');
  if (!Number.isInteger(Number(next.project.dashboard.port)) || Number(next.project.dashboard.port) <= 0) {
    next.project.dashboard.port = 8233;
  }

  next.worker_defaults = next.worker_defaults || {};
  if (!Array.isArray(next.worker_defaults.resource_pool_queue) || next.worker_defaults.resource_pool_queue.length === 0) {
    next.worker_defaults.resource_pool_queue = deriveDefaultPoolQueue(next, data);
  }
  next.worker_defaults.environment_type = firstMeaningfulValue(
    next.worker_defaults.environment_type,
    inferRuntimeWorkerValue(data, 'environment_type', true),
    'uv',
  );
  if (next.worker_defaults.environment_type !== 'none') {
    next.worker_defaults.environment_path = firstMeaningfulValue(
      next.worker_defaults.environment_path,
      inferRuntimeWorkerValue(data, 'environment_path'),
      deriveDefaultEnvironmentPath(next),
    );
  }
  next.worker_defaults.sync_command = firstMeaningfulCommand(
    next.worker_defaults.sync_command,
    inferRuntimeWorkerValue(data, 'sync_command'),
    next.worker_defaults.environment_type === 'uv' ? 'uv sync' : '',
  );
  next.worker_defaults.submit_strategy = firstMeaningfulCommand(
    next.worker_defaults.submit_strategy,
    inferRuntimeWorkerValue(data, 'submit_strategy'),
    'patch_handoff',
  );
  next.worker_defaults.test_command = firstMeaningfulCommand(
    next.worker_defaults.test_command,
    inferRuntimeWorkerValue(data, 'test_command'),
    'uv run pytest tests/moe_test.py',
  );

  const plannedWorkers = buildPlannedWorkers(data, next);
  const plannedAgents = new Set(plannedWorkers.map((worker) => worker.agent));
  const existingByAgent = new Map((next.workers || []).filter((worker) => normalizedText(worker.agent)).map((worker) => [worker.agent, worker]));

  const hydratedPlannedWorkers = plannedWorkers.map((plannedWorker) => {
    const existing = existingByAgent.get(plannedWorker.agent);
    const runtimeWorker = runtimeByAgent.get(plannedWorker.agent);
    const resolvedWorker = resolvedByAgent.get(plannedWorker.agent);
    const gitIdentity = existing?.git_identity && (existing.git_identity.name || existing.git_identity.email)
      ? existing.git_identity
      : undefined;
    return {
      agent: plannedWorker.agent,
      task_id: firstMeaningfulValue(existing?.task_id, resolvedWorker?.task_id, plannedWorker.task_id),
      branch: firstMeaningfulValue(existing?.branch, resolvedWorker?.branch, runtimeWorker?.branch, plannedWorker.branch),
      worktree_path: firstMeaningfulPath(existing?.worktree_path, resolvedWorker?.worktree_path, runtimeWorker?.worktree_path, plannedWorker.worktree_path),
      resource_pool: firstMeaningfulValue(existing?.resource_pool, resolvedWorker?.resource_pool, resolvedWorker?.locked_pool, runtimeWorker?.resource_pool),
      resource_pool_queue: Array.isArray(existing?.resource_pool_queue) && existing.resource_pool_queue.length > 0
        ? existing.resource_pool_queue
        : (resolvedWorker?.resource_pool_queue || []),
      environment_type: firstMeaningfulValue(existing?.environment_type, runtimeWorker?.environment_type),
      environment_path: firstMeaningfulPath(existing?.environment_path, runtimeWorker?.environment_path),
      sync_command: firstMeaningfulCommand(existing?.sync_command, runtimeWorker?.sync_command),
      test_command: firstMeaningfulCommand(existing?.test_command, resolvedWorker?.test_command, runtimeWorker?.test_command),
      submit_strategy: firstMeaningfulCommand(existing?.submit_strategy, runtimeWorker?.submit_strategy),
      git_identity: gitIdentity,
    };
  });

  const extraWorkers = (next.workers || [])
    .filter((worker) => !plannedAgents.has(worker.agent))
    .map((worker) => ({
      ...worker,
      branch: worker.branch || (worker.agent ? deriveBranchName(worker.agent, worker.task_id || worker.agent, worker.task_id || worker.agent) : ''),
      worktree_path: !normalizedText(worker.worktree_path) || isPlaceholderPath(worker.worktree_path) ? deriveWorktreePath(next, worker.agent) : worker.worktree_path,
    }));

  next.workers = [...hydratedPlannedWorkers, ...extraWorkers];
  return normalizeDerivedPaths(next);
}

function sectionMatchesField(section: ConfigSection, field: string): boolean {
  if (section === 'project') {
    return field.startsWith('project.') && !field.startsWith('project.integration_branch') && !field.startsWith('project.manager_git_identity');
  }
  if (section === 'merge_policy') {
    return field.startsWith('project.integration_branch') || field.startsWith('project.manager_git_identity');
  }
  if (section === 'resource_pools') {
    return field.startsWith('resource_pools.');
  }
  if (section === 'worker_defaults') {
    return field.startsWith('worker_defaults.');
  }
  return field.startsWith('workers[');
}

function collectSectionIssues(section: ConfigSection, issues: ValidationIssue[]): ValidationIssue[] {
  return issues.filter((issue) => sectionMatchesField(section, issue.field));
}

function sectionRouteUnavailable(error: unknown, path: string): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (message.includes(path) && message.includes('status 404')) || message.includes(`unknown api route: ${path}`);
}

function buildPlannedWorkers(data: DashboardState | null, config: ConfigShape): PlannedWorker[] {
  if (!data) {
    return [];
  }

  const byAgent = new Map<string, PlannedWorker>();
  (data.backlog.items || []).forEach((item) => {
    const agent = String(item.owner || '').trim();
    if (!agent || agent === 'A0') {
      return;
    }
    if (!byAgent.has(agent)) {
      byAgent.set(agent, {
        agent,
        task_id: item.id,
        title: item.title,
        branch: deriveBranchName(agent, item.title, item.id),
        worktree_path: deriveWorktreePath(config, agent),
      });
    }
  });

  (data.runtime.workers || []).forEach((item) => {
    const agent = String(item.agent || '').trim();
    if (!agent || agent === 'A0' || byAgent.has(agent)) {
      return;
    }
    byAgent.set(agent, {
      agent,
      task_id: `${agent}-001`,
      title: item.branch || agent,
      branch: item.branch || deriveBranchName(agent, item.branch || agent, `${agent}-001`),
      worktree_path: deriveWorktreePath(config, agent),
    });
  });

  return Array.from(byAgent.values()).sort((left, right) => left.agent.localeCompare(right.agent, undefined, { numeric: true }));
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

function getLocalValidationIssues(config: ConfigShape, data: DashboardState | null): ValidationIssue[] {
  const draft = hydrateConfigForA0(data, normalizeConfig(config));
  const issues: ValidationIssue[] = [];
  const add = (field: string, message: string) => issues.push({ field, message });
  const project = draft.project || {};
  const dashboard = project.dashboard || {};
  const pools = draft.resource_pools || {};
  const workerDefaults = draft.worker_defaults || {};
  const workers = draft.workers || [];

  if (!String(project.repository_name || '').trim()) {
    add('project.repository_name', 'repository name is required');
  }
  if (!String(project.local_repo_root || '').trim()) {
    add('project.local_repo_root', 'local repo root is required');
  }
  const referenceWorkspace = projectReferenceWorkspace(project);
  if (referenceWorkspace && referenceWorkspace.startsWith('/absolute/path/')) {
    add('project.reference_workspace_root', 'reference workspace must be replaced with a real path');
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

  if (workerDefaults.resource_pool && !pools[workerDefaults.resource_pool]) {
    add('worker_defaults.resource_pool', 'default resource pool must refer to an existing pool');
  }
  if (workerDefaults.resource_pool_queue && workerDefaults.resource_pool_queue.some((poolName) => !pools[poolName])) {
    add('worker_defaults.resource_pool_queue', 'default queue must contain only existing pools');
  }
  if (workerDefaults.git_identity?.name && !workerDefaults.git_identity?.email) {
    add('worker_defaults.git_identity.email', 'default git identity email is required when name is set');
  }
  if (workerDefaults.git_identity?.email && !workerDefaults.git_identity?.name) {
    add('worker_defaults.git_identity.name', 'default git identity name is required when email is set');
  }

  workers.forEach((worker, index) => {
    const effectiveWorker = mergeWorkerWithDefaults(worker, workerDefaults);
    const root = `workers[${index}]`;
    const agent = String(worker.agent || '').trim();
    const branch = String(effectiveWorker.branch || '').trim();
    const worktreePath = String(effectiveWorker.worktree_path || '').trim();
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
    const poolName = String(effectiveWorker.resource_pool || '').trim();
    const queue = effectiveWorker.resource_pool_queue || [];
    if (!poolName && !queue.length) {
      add(`${root}.resource_pool`, 'resource pool or queue is required');
    }
    if (!String(effectiveWorker.test_command || '').trim()) {
      add(`${root}.test_command`, 'test command is required');
    }
    if (!String(effectiveWorker.submit_strategy || '').trim()) {
      add(`${root}.submit_strategy`, 'submit strategy is required');
    }
    if (String(effectiveWorker.environment_type || 'uv') !== 'none' && !String(effectiveWorker.environment_path || '').trim()) {
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
  helpText,
  placeholder,
  type = 'text',
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  issues?: string[];
  helpText?: string;
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
      {helpText ? <span className="field-help">{helpText}</span> : null}
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

function SectionHeader({
  title,
  section,
  status,
  onValidate,
  onSave,
  action,
  subtitle,
}: {
  title: string;
  section: ConfigSection;
  status?: { message: string; error: boolean };
  onValidate: (section: ConfigSection) => void;
  onSave: (section: ConfigSection) => void;
  action?: ReactNode;
  subtitle?: ReactNode;
}) {
  return (
    <div className="section-head section-head-actions">
      <div>
        <h3>{title}</h3>
        {subtitle ? <div className="section-subtitle small muted">{subtitle}</div> : null}
      </div>
      <div className="section-actions">
        {status?.message ? <span className={classNames('section-status', status.error && 'error')}>{status.message}</span> : null}
        <button className="ghost" type="button" onClick={() => onValidate(section)}>Validate</button>
        <button type="button" onClick={() => onSave(section)}>Save</button>
        {action}
      </div>
    </div>
  );
}

function AutomationSummary({ draftConfig, data }: { draftConfig: ConfigShape; data: DashboardState }) {
  const plannedWorkers = buildPlannedWorkers(data, draftConfig);
  const autoManaged = [
    'Repository name and dashboard host/port defaults',
    'Worker roster from backlog and runtime state',
    'Worker task IDs from the current plan',
    'Suggested branch names from task ownership and title',
    'Worktree paths from repo root plus agent ID',
    'Task-aware resource-pool recommendation and lock from provider quality history',
    'Default pool queue from live provider ranking',
    'Environment type/path defaults',
    'Default sync command',
    'Default submit strategy',
    'Task-aware test-command selection with safe fallbacks',
  ];
  const userOnly = [
    'Local repo root if A0 cannot infer it correctly',
    'Reference workspace path when the project really needs one',
    'Integration branch policy',
    'Resource-pool credentials, provider choice, and model choice',
    'Only exceptional per-worker routing or environment overrides that truly differ from the default path',
  ];

  return (
    <section className="helper-card settings-card settings-card-wide">
      <div className="section-head">
        <h3>A0-Managed Defaults</h3>
        <div className="small muted">{autoManaged.length} areas auto-managed, {userOnly.length} areas still need human confirmation</div>
      </div>
      <p className="small muted">Planned workers currently detected: {plannedWorkers.map((worker) => worker.agent).join(', ') || 'none'}. In Settings, A0 plan means the derived target state; override means a human-pinned exception that should be rare and easy to reset.</p>
      <div className="automation-grid">
        <div className="subcard">
          <div className="subcard-title">You should usually not need to fill these by hand</div>
          <ul className="automation-list">
            {autoManaged.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
        <div className="subcard">
          <div className="subcard-title">You should mostly only confirm these</div>
          <ul className="automation-list">
            {userOnly.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      </div>
    </section>
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
    { key: 'reference_workspace_root', value: projectReferenceWorkspace(data.project) },
    { key: 'integration_branch', value: data.project.integration_branch || data.project.base_branch || '' },
    { key: 'dashboard', value: data.project.dashboard?.host && data.project.dashboard?.port ? `${data.project.dashboard.host}:${data.project.dashboard.port}` : '' },
    { key: 'listener_active', value: data.mode.listener_active },
  ];
  const processRows = Object.entries(data.processes || {}).map(([agent, item]) => ({ agent, provider: item.provider, model: item.model, alive: item.alive, pid: item.pid, resource_pool: item.resource_pool, recursion_guard: item.recursion_guard, wrapper_path: item.wrapper_path, returncode: item.returncode }));
  const mergeRows = data.merge_queue.map((item) => ({ agent: item.agent, branch: item.branch, submit_strategy: item.submit_strategy, worker_identity: item.worker_identity, merge_target: item.merge_target, status: item.status, manager_action: item.manager_action }));
  const providerRows = data.provider_queue.map((item) => ({ resource_pool: item.resource_pool, provider: item.provider, priority: item.priority, binary_found: item.binary_found, recursion_guard: item.recursion_guard, launch_wrapper: item.launch_wrapper, auth_mode: item.auth_mode, auth_ready: item.auth_ready, launch_ready: item.launch_ready, auth_detail: item.auth_detail, connection_quality: item.connection_quality, work_quality: item.work_quality, score: item.score }));
  return (
    <div className="tab-body">
      <section className="grid">
        <section className="card"><h2>Commands</h2><pre>{`serve:\n${data.commands.serve}\n\nup:\n${data.commands.up}`}</pre></section>
        <section className="card"><h2>Validation</h2><pre>{renderValidation(data)}</pre></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Provider Queue</h2><DataTable columns={['resource_pool', 'provider', 'priority', 'binary_found', 'recursion_guard', 'launch_wrapper', 'auth_mode', 'auth_ready', 'launch_ready', 'auth_detail', 'connection_quality', 'work_quality', 'score']} rows={providerRows} /></section>
        <section className="card"><h2>Merge Queue</h2><DataTable columns={['agent', 'branch', 'submit_strategy', 'worker_identity', 'merge_target', 'status', 'manager_action']} rows={mergeRows} /></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Active Processes</h2><DataTable columns={['agent', 'provider', 'model', 'alive', 'pid', 'resource_pool', 'recursion_guard', 'wrapper_path', 'returncode']} rows={processRows} /></section>
        <section className="card"><h2>Project</h2><DataTable columns={['key', 'value']} rows={projectRows} /></section>
      </section>
      <section className="grid">
        <section className="card"><h2>Runtime Topology</h2><DataTable columns={['agent', 'resource_pool', 'provider', 'model', 'branch', 'recursion_guard', 'launch_wrapper', 'status']} rows={data.runtime.workers || []} /></section>
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
  data,
  draftConfig,
  providerOptions,
  allIssues,
  sectionStatuses,
  onProjectChange,
  onMergeChange,
  onPoolChange,
  onAddPool,
  onWorkerChange,
  onAddWorker,
  onValidateSection,
  onSaveSection,
  onSyncWorkers,
  onAutoFillWorktreePaths,
  onResetWorkerDefaults,
  onResetWorkerOverrides,
}: {
  data: DashboardState;
  draftConfig: ConfigShape;
  providerOptions: string[];
  allIssues: ValidationIssue[];
  sectionStatuses: SectionStatusMap;
  onProjectChange: (field: string, value: string) => void;
  onMergeChange: (field: string, value: string) => void;
  onPoolChange: (poolName: string, field: keyof ConfigResourcePool, value: string) => void;
  onAddPool: () => void;
  onWorkerChange: (index: number, field: string, value: string) => void;
  onAddWorker: () => void;
  onValidateSection: (section: ConfigSection) => void;
  onSaveSection: (section: ConfigSection) => void;
  onSyncWorkers: () => void;
  onAutoFillWorktreePaths: () => void;
  onResetWorkerDefaults: () => void;
  onResetWorkerOverrides: (index: number, scope?: WorkerResetScope) => void;
}) {
  const project = draftConfig.project || {};
  const dashboard = project.dashboard || {};
  const pools = draftConfig.resource_pools || {};
  const workerDefaults = draftConfig.worker_defaults || {};
  const workers = draftConfig.workers || [];
  const plannedWorkers = buildPlannedWorkers(data, draftConfig);
  const resolvedByAgent = buildResolvedWorkerMap(data);
  const issues = buildIssueMap(allIssues);
  return (
    <div className="tab-body">
      <section className="card">
        <div className="page-header">
          <div>
            <h2>Settings</h2>
            <p className="small">A0 now auto-hydrates the worker roster, branch proposals, worktree paths, and shared defaults. You should mostly verify project paths, pool routing, and true exceptions.</p>
          </div>
        </div>
        <div className="settings-stack">
          <section className="helper-card settings-card settings-card-wide">
            <SectionHeader title="Resource Pools" section="resource_pools" status={sectionStatuses.resource_pools} onValidate={onValidateSection} onSave={onSaveSection} action={<button className="ghost" type="button" onClick={onAddPool}>Add Pool</button>} />
            <SectionIssueList issues={collectSectionIssues('resource_pools', allIssues)} />
            <div className="pool-strip">
              {Object.entries(pools).map(([poolName, pool]) => (
                <div key={poolName} className="subcard pool-card">
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

          <div className="settings-duo">
            <section className="helper-card settings-card">
            <SectionHeader title="Project" section="project" status={sectionStatuses.project} onValidate={onValidateSection} onSave={onSaveSection} />
            <SectionIssueList issues={collectSectionIssues('project', allIssues)} />
            <div className="field-grid">
              <Field label="Repository" value={project.repository_name || ''} onChange={(value) => onProjectChange('repository_name', value)} issues={issues['project.repository_name']} />
              <Field label="Local Repo Root" value={project.local_repo_root || ''} onChange={(value) => onProjectChange('local_repo_root', value)} issues={issues['project.local_repo_root']} />
              <Field label="Reference Workspace" value={projectReferenceWorkspace(project)} onChange={(value) => onProjectChange('reference_workspace_root', value)} issues={issues['project.reference_workspace_root']} helpText="Optional shared reference repo or baseline workspace for task guidance." />
              <Field label="Dashboard Host" value={dashboard.host || ''} onChange={(value) => onProjectChange('dashboard.host', value)} issues={issues['project.dashboard.host']} />
              <Field label="Dashboard Port" type="number" value={dashboard.port || 8233} onChange={(value) => onProjectChange('dashboard.port', value)} issues={issues['project.dashboard.port']} />
            </div>
          </section>

            <section className="helper-card settings-card">
            <SectionHeader title="Merge Policy" section="merge_policy" status={sectionStatuses.merge_policy} onValidate={onValidateSection} onSave={onSaveSection} />
            <SectionIssueList issues={collectSectionIssues('merge_policy', allIssues)} />
            <div className="field-grid">
              <Field label="Integration Branch" value={project.integration_branch || project.base_branch || ''} onChange={(value) => onMergeChange('integration_branch', value)} issues={issues['project.integration_branch']} />
              <Field label="Manager Name" value={project.manager_git_identity?.name || ''} onChange={(value) => onMergeChange('manager_git_identity.name', value)} />
              <Field label="Manager Email" value={project.manager_git_identity?.email || ''} onChange={(value) => onMergeChange('manager_git_identity.email', value)} />
            </div>
          </section>
          </div>

          <section className="helper-card settings-card settings-card-wide">
            <SectionHeader
              title="Worker Defaults"
              section="worker_defaults"
              status={sectionStatuses.worker_defaults}
              onValidate={onValidateSection}
              onSave={onSaveSection}
              subtitle="Common defaults are the few knobs you may actually standardize across workers. Advanced defaults are fallback overrides for exceptional environments."
              action={<button className="ghost" type="button" onClick={onResetWorkerDefaults}>Reset to A0</button>}
            />
            <SectionIssueList issues={collectSectionIssues('worker_defaults', allIssues)} />
            <p className="small muted">These values apply to every worker unless a row below overrides them. Blank fields are auto-filled from runtime conventions or sensible defaults where possible, so the main path should stay sparse.</p>
            <div className="field-grid compact-field-grid">
              <Field label="Default Pool" value={workerDefaults.resource_pool || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.resource_pool', value)} issues={issues['worker_defaults.resource_pool']} helpText="Leave blank to rely on pool queue or per-worker overrides." />
              <Field label="Default Pool Queue" value={stringifyQueue(workerDefaults.resource_pool_queue)} onChange={(value) => onWorkerChange(-1, 'worker_defaults.resource_pool_queue', value)} issues={issues['worker_defaults.resource_pool_queue']} placeholder="copilot_pool, claude_pool" />
              <SelectField label="Default Environment" value={workerDefaults.environment_type || 'uv'} onChange={(value) => onWorkerChange(-1, 'worker_defaults.environment_type', value)} options={['uv', 'venv', 'none']} />
              <Field label="Default Environment Path" value={workerDefaults.environment_path || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.environment_path', value)} issues={issues['worker_defaults.environment_path']} />
            </div>
            <details className="advanced-panel defaults-panel">
              <summary>Advanced defaults</summary>
              <div className="field-grid advanced-grid">
                <Field label="Default Sync Command" value={workerDefaults.sync_command || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.sync_command', value)} helpText="Leave blank to let A0 follow the environment convention." />
                <Field label="Default Test Command" value={workerDefaults.test_command || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.test_command', value)} issues={issues['worker_defaults.test_command']} helpText="Leave blank to let task policy choose per worker." />
                <Field label="Default Submit Strategy" value={workerDefaults.submit_strategy || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.submit_strategy', value)} issues={issues['worker_defaults.submit_strategy']} helpText="Leave blank to keep A0's standard handoff flow." />
                <Field label="Default Git Name" value={workerDefaults.git_identity?.name || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.git_identity.name', value)} issues={issues['worker_defaults.git_identity.name']} />
                <Field label="Default Git Email" value={workerDefaults.git_identity?.email || ''} onChange={(value) => onWorkerChange(-1, 'worker_defaults.git_identity.email', value)} issues={issues['worker_defaults.git_identity.email']} />
              </div>
            </details>
          </section>

          <AutomationSummary draftConfig={draftConfig} data={data} />

          <section className="helper-card settings-card settings-card-wide">
            <SectionHeader
              title="Worker Config"
              section="workers"
              status={sectionStatuses.workers}
              onValidate={onValidateSection}
              onSave={onSaveSection}
              action={
                <>
                  <button className="ghost" type="button" onClick={onSyncWorkers}>Sync From Plan</button>
                  <button className="ghost" type="button" onClick={onAutoFillWorktreePaths}>Auto Paths</button>
                  <button className="ghost" type="button" onClick={onAddWorker}>Add Worker</button>
                </>
              }
            />
            <SectionIssueList issues={collectSectionIssues('workers', allIssues)} />
            <p className="small muted">Detected workers from backlog/runtime: {plannedWorkers.map((item) => item.agent).join(', ') || 'none'}. A0 plan is the derived execution target. Any filled override below becomes a human-pinned exception; use Reset to A0 to clear those pins and fall back to the plan.</p>
            <div className="worker-grid">
              {workers.map((worker, index) => (
                <div key={`${worker.agent || 'worker'}-${index}`} className="subcard">
                  {(() => {
                    const resolved = resolvedByAgent.get(worker.agent || '');
                    const planned = workerPlanView(worker, draftConfig, data, plannedWorkers);
                    const recommendation = planned.poolReason || '';
                    const suggestedTest = planned.suggestedTestCommand || '';
                    return (
                      <>
                  <div className="subcard-title worker-card-title-row">
                    <span>{worker.agent || `Worker ${index + 1}`}</span>
                    <button className="ghost" type="button" onClick={() => onResetWorkerOverrides(index)}>Reset to A0</button>
                  </div>
                  {recommendation ? <p className="small muted">{recommendation}</p> : null}
                  <div className="plan-grid">
                    <div className="plan-row"><span className="muted">Task</span><strong>{planned.taskId || 'A0 will assign'}</strong></div>
                    <div className="plan-row"><span className="muted">Type</span><strong>{planned.taskType || 'default'}</strong></div>
                    <div className="plan-row"><span className="muted">Branch</span><strong>{planned.branch || 'A0 will derive'}</strong></div>
                    <div className="plan-row"><span className="muted">Worktree</span><strong>{planned.worktreePath || 'derived from Local Repo Root'}</strong></div>
                    <div className="plan-row"><span className="muted">Pool</span><strong>{planned.lockedPool || planned.recommendedPool || 'A0 routing'}</strong></div>
                    <div className="plan-row"><span className="muted">Test</span><strong>{planned.testCommand || suggestedTest || 'A0 default'}</strong></div>
                  </div>
                  <div className="field-grid compact-field-grid">
                    <Field label="Agent" value={worker.agent || ''} onChange={(value) => onWorkerChange(index, 'agent', value)} issues={issues[`workers[${index}].agent`]} />
                    <Field label="Pool Override" value={worker.resource_pool || ''} onChange={(value) => onWorkerChange(index, 'resource_pool', value)} issues={issues[`workers[${index}].resource_pool`]} helpText={resolved?.locked_pool ? `A0 lock: ${resolved.locked_pool}` : resolved?.recommended_pool ? `A0 recommends: ${resolved.recommended_pool}` : (workerDefaults.resource_pool ? `Default: ${workerDefaults.resource_pool}` : 'Blank means inherit A0 routing.')} />
                  </div>
                  <details className="advanced-panel">
                    <summary>Advanced overrides</summary>
                    <div className="override-toolbar">
                      <button className="ghost" type="button" onClick={() => onResetWorkerOverrides(index, 'routing')}>Reset routing</button>
                      <button className="ghost" type="button" onClick={() => onResetWorkerOverrides(index, 'runtime')}>Reset runtime</button>
                    </div>
                    <div className="field-grid advanced-grid">
                      <Field label="Task ID Override" value={worker.task_id || ''} onChange={(value) => onWorkerChange(index, 'task_id', value)} helpText={planned.taskId ? `A0 plan: ${planned.taskId}` : 'Leave blank to inherit A0 task assignment.'} />
                      <Field label="Branch Override" value={worker.branch || ''} onChange={(value) => onWorkerChange(index, 'branch', value)} issues={issues[`workers[${index}].branch`]} helpText={planned.branch ? `A0 plan: ${planned.branch}` : 'Leave blank to let A0 derive the branch.'} />
                      <Field label="Worktree Path Override" value={worker.worktree_path || ''} onChange={(value) => onWorkerChange(index, 'worktree_path', value)} issues={issues[`workers[${index}].worktree_path`]} helpText={planned.worktreePath ? `A0 plan: ${planned.worktreePath}` : 'Leave blank to derive from Local Repo Root.'} />
                      <Field label="Queue Override" value={stringifyQueue(worker.resource_pool_queue)} onChange={(value) => onWorkerChange(index, 'resource_pool_queue', value)} issues={issues[`workers[${index}].resource_pool_queue`]} placeholder="pool_a, pool_b" helpText={resolved?.resource_pool_queue?.length ? `A0 order: ${stringifyQueue(resolved.resource_pool_queue)}` : (workerDefaults.resource_pool_queue?.length ? `Default: ${stringifyQueue(workerDefaults.resource_pool_queue)}` : 'Blank means inherit A0 queue.')} />
                      <SelectField label="Environment Type" value={worker.environment_type || ''} onChange={(value) => onWorkerChange(index, 'environment_type', value)} options={['uv', 'venv', 'none']} />
                      <Field label="Environment Path" value={worker.environment_path || ''} onChange={(value) => onWorkerChange(index, 'environment_path', value)} issues={issues[`workers[${index}].environment_path`]} helpText={workerDefaults.environment_path ? `Default: ${workerDefaults.environment_path}` : undefined} />
                      <Field label="Sync Command" value={worker.sync_command || ''} onChange={(value) => onWorkerChange(index, 'sync_command', value)} helpText={workerDefaults.sync_command ? `Default: ${workerDefaults.sync_command}` : undefined} />
                      <Field label="Test Command" value={worker.test_command || ''} onChange={(value) => onWorkerChange(index, 'test_command', value)} issues={issues[`workers[${index}].test_command`]} helpText={suggestedTest ? `A0 picked: ${suggestedTest}` : (workerDefaults.test_command ? `Default: ${workerDefaults.test_command}` : undefined)} />
                      <Field label="Submit Strategy" value={worker.submit_strategy || ''} onChange={(value) => onWorkerChange(index, 'submit_strategy', value)} issues={issues[`workers[${index}].submit_strategy`]} helpText={workerDefaults.submit_strategy ? `Default: ${workerDefaults.submit_strategy}` : undefined} />
                      <Field label="Git Name" value={worker.git_identity?.name || ''} onChange={(value) => onWorkerChange(index, 'git_identity.name', value)} helpText={workerDefaults.git_identity?.name ? `Default: ${workerDefaults.git_identity.name}` : undefined} />
                      <Field label="Git Email" value={worker.git_identity?.email || ''} onChange={(value) => onWorkerChange(index, 'git_identity.email', value)} helpText={workerDefaults.git_identity?.email ? `Default: ${workerDefaults.git_identity.email}` : undefined} />
                    </div>
                  </details>
                      </>
                    );
                  })()}
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
  const [draftConfig, setDraftConfig] = useState<ConfigShape>({ project: {}, providers: {}, resource_pools: {}, worker_defaults: {}, workers: [] });
  const [configDirty, setConfigDirty] = useState(false);
  const [launchStrategy, setLaunchStrategy] = useState<LaunchStrategy>('initial_copilot');
  const [launchProvider, setLaunchProvider] = useState('copilot');
  const [launchModel, setLaunchModel] = useState('');
  const [launchDirty, setLaunchDirty] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [actionInFlight, setActionInFlight] = useState(false);
  const [status, setStatus] = useState<{ message: string; error: boolean }>({ message: '', error: false });
  const [backendIssues, setBackendIssues] = useState<ValidationIssue[]>([]);
  const [sectionStatuses, setSectionStatuses] = useState<SectionStatusMap>({});
  const abortRef = useRef<AbortController | null>(null);

  const agentRows = useMemo(() => buildAgentRows(data), [data]);
  const progress = useMemo(() => buildProgressModel(data, agentRows), [data, agentRows]);
  const localIssues = useMemo(() => getLocalValidationIssues(draftConfig, data), [draftConfig, data]);
  const allIssues = useMemo(() => [...localIssues, ...backendIssues], [localIssues, backendIssues]);
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
        setDraftConfig(hydrateConfigForA0(nextData, cloneConfig(nextData.config)));
        setBackendIssues([]);
        setSectionStatuses({});
      }
      if (!launchDirty) {
        setLaunchStrategy(nextData.launch_policy.default_strategy);
        setLaunchProvider(preferredLaunchProvider(nextData.launch_policy));
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
    setDraftConfig((current) => hydrateConfigForA0(data, normalizeConfig(updater(normalizeConfig(cloneConfig(current))))));
  };

  const refreshStateOnly = async () => {
    const controller = new AbortController();
    const nextData = await fetchState(controller.signal);
    setData(nextData);
    return nextData;
  };

  const onProjectChange = (field: string, value: string) => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const previousAutoEnvPath = deriveDefaultEnvironmentPath(current);
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
      } else if (field === 'reference_workspace_root' || field === 'paddle_repo_path') {
        next.project.reference_workspace_root = value;
      }
      const nextAutoEnvPath = deriveDefaultEnvironmentPath(next);
      next.worker_defaults = next.worker_defaults || {};
      if (next.worker_defaults.environment_type !== 'none') {
        const currentEnvironmentPath = normalizedText(next.worker_defaults.environment_path);
        if (!currentEnvironmentPath || isPlaceholderPath(currentEnvironmentPath) || currentEnvironmentPath === previousAutoEnvPath) {
          next.worker_defaults.environment_path = nextAutoEnvPath;
        }
      }
      next.workers = (next.workers || []).map((worker) => ({
        ...worker,
        worktree_path: (() => {
          const previousDerived = deriveWorktreePath(current, worker.agent);
          const nextDerived = deriveWorktreePath(next, worker.agent);
          const currentPath = normalizedText(worker.worktree_path);
          return !currentPath || isPlaceholderPath(currentPath) || currentPath === previousDerived ? nextDerived : worker.worktree_path;
        })(),
      }));
      return normalizeDerivedPaths(next);
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
      if (index === -1) {
        next.worker_defaults = next.worker_defaults || {};
        if (field === 'worker_defaults.resource_pool_queue') {
          next.worker_defaults.resource_pool_queue = parseQueue(value);
        } else if (field === 'worker_defaults.git_identity.name' || field === 'worker_defaults.git_identity.email') {
          next.worker_defaults.git_identity = next.worker_defaults.git_identity || {};
          if (field.endsWith('.name')) {
            next.worker_defaults.git_identity.name = value;
          } else {
            next.worker_defaults.git_identity.email = value;
          }
        } else {
          const normalizedField = field.replace('worker_defaults.', '');
          (next.worker_defaults as Record<string, unknown>)[normalizedField] = value;
        }
        return next;
      }
      const workers = [...(next.workers || [])];
      const worker = { ...(workers[index] || { agent: `A${index + 1}` }) };
      const previousAgent = String(worker.agent || '');
      const previousTaskId = String(worker.task_id || '');
      const previousBranch = String(worker.branch || '');
      const previousWorktreePath = String(worker.worktree_path || '');
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
      if (field === 'agent') {
        const nextAgent = String(value || '');
        const previousDerivedTaskId = previousAgent ? `${previousAgent}-001` : '';
        const nextDerivedTaskId = nextAgent ? `${nextAgent}-001` : '';
        const previousDerivedBranch = previousAgent
          ? deriveBranchName(previousAgent, previousTaskId || previousAgent, previousTaskId || previousAgent)
          : '';
        const nextDerivedBranch = nextAgent
          ? deriveBranchName(nextAgent, worker.task_id || nextAgent, worker.task_id || nextAgent)
          : '';
        const previousDerivedWorktreePath = previousAgent ? deriveWorktreePath(next, previousAgent) : '';
        const nextDerivedWorktreePath = nextAgent ? deriveWorktreePath(next, nextAgent) : '';

        if (!previousTaskId || previousTaskId === previousDerivedTaskId) {
          worker.task_id = nextDerivedTaskId;
        }
        if (!previousBranch || previousBranch === previousDerivedBranch) {
          worker.branch = nextDerivedBranch;
        }
        if (!previousWorktreePath || previousWorktreePath === previousDerivedWorktreePath) {
          worker.worktree_path = nextDerivedWorktreePath;
        }
      }
      workers[index] = worker;
      next.workers = workers;
      return normalizeDerivedPaths(next);
    });
  };

  const onAddWorker = () => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const workers = [...(next.workers || [])];
      const usedAgents = new Set(workers.map((worker) => worker.agent));
      const plannedCandidate = buildPlannedWorkers(data, next).find((worker) => !usedAgents.has(worker.agent));
      if (plannedCandidate) {
        workers.push({
          agent: plannedCandidate.agent,
          task_id: plannedCandidate.task_id,
          resource_pool: '',
          resource_pool_queue: [],
          worktree_path: plannedCandidate.worktree_path,
          branch: plannedCandidate.branch,
        });
      } else {
        const nextAgent = `A${workers.length + 1}`;
        workers.push({
          agent: nextAgent,
          task_id: `${nextAgent}-001`,
          resource_pool: '',
          resource_pool_queue: [],
          worktree_path: deriveWorktreePath(next, nextAgent),
          branch: deriveBranchName(nextAgent, nextAgent, `${nextAgent}-001`),
        });
      }
      next.workers = workers;
      return normalizeDerivedPaths(next);
    });
  };

  const onSyncWorkers = () => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const plannedWorkers = buildPlannedWorkers(data, next);
      const existingByAgent = new Map((next.workers || []).map((worker) => [worker.agent, worker]));
      const syncedWorkers = plannedWorkers.map((plannedWorker) => {
        const existing = existingByAgent.get(plannedWorker.agent);
        return {
          agent: plannedWorker.agent,
          task_id: existing?.task_id || plannedWorker.task_id,
          branch: existing?.branch || plannedWorker.branch,
          worktree_path: existing?.worktree_path || plannedWorker.worktree_path,
          resource_pool: existing?.resource_pool || '',
          resource_pool_queue: existing?.resource_pool_queue || [],
          environment_type: existing?.environment_type,
          environment_path: existing?.environment_path,
          sync_command: existing?.sync_command,
          test_command: existing?.test_command,
          submit_strategy: existing?.submit_strategy,
          git_identity: existing?.git_identity,
        };
      });
      const extraWorkers = (next.workers || []).filter((worker) => !plannedWorkers.some((plannedWorker) => plannedWorker.agent === worker.agent));
      next.workers = [...syncedWorkers, ...extraWorkers];
      return normalizeDerivedPaths(next);
    });
    setStampedStatus('worker list synced from backlog and runtime plan');
  };

  const onAutoFillWorktreePaths = () => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      next.workers = (next.workers || []).map((worker) => ({
        ...worker,
        worktree_path: !normalizedText(worker.worktree_path) || isPlaceholderPath(worker.worktree_path) ? deriveWorktreePath(next, worker.agent) : worker.worktree_path,
      }));
      return normalizeDerivedPaths(next);
    });
    setStampedStatus('missing worktree paths filled from local repo root');
  };

  const onResetWorkerDefaults = () => {
    updateConfig((current) => ({
      ...normalizeConfig(current),
      worker_defaults: resetWorkerDefaultsToA0(current.worker_defaults),
    }));
    setStampedStatus('worker defaults reset to A0-managed defaults');
  };

  const onResetWorkerOverrides = (index: number, scope: WorkerResetScope = 'all') => {
    updateConfig((current) => {
      const next = normalizeConfig(current);
      const workers = [...(next.workers || [])];
      if (!workers[index]) {
        return next;
      }
      workers[index] = resetWorkerOverridesToA0(workers[index], scope);
      next.workers = workers;
      return next;
    });
    const message = scope === 'all'
      ? 'worker overrides reset to A0 plan'
      : scope === 'routing'
        ? 'worker routing overrides cleared'
        : 'worker runtime overrides cleared';
    setStampedStatus(message);
  };

  const onValidateSection = (section: ConfigSection) => void runAction(`validating ${section}`, async () => {
    const effectiveDraft = section === 'project'
      ? configForProjectSave(data?.config, draftConfig)
      : normalizeDerivedPaths(draftConfig);
    const sectionValue = buildSectionValue(effectiveDraft, section);
    let validation;
    let usedFullConfigFallback = section === 'project';
    try {
      validation = usedFullConfigFallback ? await validateConfig(effectiveDraft) : await validateConfigSection(section, sectionValue);
    } catch (error) {
      if (!sectionRouteUnavailable(error, '/api/config/validate-section')) {
        throw error;
      }
      validation = await validateConfig(effectiveDraft);
      usedFullConfigFallback = true;
    }
    const nextIssues = usedFullConfigFallback
      ? collectSectionIssues(section, validation.validation_issues)
      : validation.validation_issues;
    const blockingOtherIssues = usedFullConfigFallback
      ? validation.validation_issues.filter((issue) => !sectionMatchesField(section, issue.field)).length
      : 0;
    setBackendIssues((current) => [
      ...current.filter((issue) => usedFullConfigFallback || !sectionMatchesField(section, issue.field)),
      ...(usedFullConfigFallback ? validation.validation_issues : nextIssues),
    ]);
    setSectionStatuses((current) => ({
      ...current,
      [section]: {
        message: validation.ok
          ? (usedFullConfigFallback ? 'validated via full-config fallback' : 'validated')
          : usedFullConfigFallback && blockingOtherIssues > 0
            ? `${nextIssues.length} section issue(s), ${blockingOtherIssues} other blocker(s)`
            : `${nextIssues.length} issue(s)`,
        error: !validation.ok,
      },
    }));
    setStampedStatus(
      validation.ok
        ? `${section} validated${usedFullConfigFallback ? ' via full-config fallback' : ''}`
        : `${section} has validation issues`,
      !validation.ok,
    );
  });

  const onSaveSection = (section: ConfigSection) => void runAction(`saving ${section}`, async () => {
    const sectionLocalIssues = collectSectionIssues(section, localIssues);
    if (sectionLocalIssues.length > 0) {
      setSectionStatuses((current) => ({
        ...current,
        [section]: { message: `${sectionLocalIssues.length} local issue(s)`, error: true },
      }));
      setStampedStatus(`${section} contains local validation issues`, true);
      return;
    }
    const effectiveDraft = section === 'project'
      ? configForProjectSave(data?.config, draftConfig)
      : normalizeDerivedPaths(draftConfig);
    const sectionValue = buildSectionValue(effectiveDraft, section);
    let validation;
    let usedFullConfigFallback = section === 'project';
    try {
      validation = usedFullConfigFallback ? await validateConfig(effectiveDraft) : await validateConfigSection(section, sectionValue);
    } catch (error) {
      if (!sectionRouteUnavailable(error, '/api/config/validate-section')) {
        throw error;
      }
      validation = await validateConfig(effectiveDraft);
      usedFullConfigFallback = true;
    }
    const validationIssues = usedFullConfigFallback
      ? validation.validation_issues
      : validation.validation_issues;
    const sectionIssues = collectSectionIssues(section, validationIssues);
    const blockingOtherIssues = usedFullConfigFallback
      ? validationIssues.filter((issue) => !sectionMatchesField(section, issue.field)).length
      : 0;
    if (!validation.ok) {
      setBackendIssues((current) => [
        ...current.filter((issue) => usedFullConfigFallback || !sectionMatchesField(section, issue.field)),
        ...validationIssues,
      ]);
      setSectionStatuses((current) => ({
        ...current,
        [section]: {
          message: usedFullConfigFallback && blockingOtherIssues > 0
            ? `${sectionIssues.length} section issue(s), ${blockingOtherIssues} other blocker(s)`
            : `${sectionIssues.length} issue(s)`,
          error: true,
        },
      }));
      setStampedStatus(`${section} rejected by validation`, true);
      return;
    }
    const response = usedFullConfigFallback
      ? await saveConfig(effectiveDraft)
      : await saveConfigSection(section, sectionValue);
    setBackendIssues((current) => current.filter((issue) => usedFullConfigFallback || !sectionMatchesField(section, issue.field)));
    setSectionStatuses((current) => ({
      ...current,
      [section]: { message: usedFullConfigFallback ? 'saved via full-config fallback' : 'saved', error: false },
    }));
    const nextData = await refreshStateOnly();
    setDraftConfig((current) => {
      const mergedDraft = hydrateConfigForA0(nextData, mergeSavedSection(current, nextData.config, section));
      setConfigDirty(JSON.stringify(normalizeConfig(mergedDraft)) !== JSON.stringify(normalizeConfig(nextData.config)));
      return mergedDraft;
    });
    setStampedStatus(
      `${section} saved${usedFullConfigFallback ? ' via full-config fallback' : ''}: ${response.validation_errors.length} note(s), ${response.launch_blockers.length} blocker(s)`,
    );
  });

  const onLaunch = (restart: boolean) => void runAction(restart ? 'restarting workers' : 'launching workers', async () => {
    try {
      const response = await launchWorkers(restart, {
        strategy: launchStrategy,
        provider: launchStrategy === 'elastic' ? undefined : launchProvider,
        model: launchStrategy === 'selected_model' ? launchModel : undefined,
      });
      setStampedStatus(
        `launch complete (${launchStrategyLabel(response.launch_policy?.strategy || launchStrategy)}): ${(response.launched || []).length} launched, ${(response.failures || []).length} failures`,
        !response.ok,
      );
    } catch (error) {
      throw new Error(formatLaunchErrorMessage(error));
    } finally {
      await refresh(true);
    }
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
                          setLaunchProvider(preferredLaunchProvider(data.launch_policy));
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
                  data={data}
                  draftConfig={draftConfig}
                  providerOptions={providerOptions}
                  allIssues={allIssues}
                  sectionStatuses={sectionStatuses}
                  onProjectChange={onProjectChange}
                  onMergeChange={onMergeChange}
                  onPoolChange={onPoolChange}
                  onAddPool={onAddPool}
                  onWorkerChange={onWorkerChange}
                  onAddWorker={onAddWorker}
                  onValidateSection={onValidateSection}
                  onSaveSection={onSaveSection}
                  onSyncWorkers={onSyncWorkers}
                  onAutoFillWorktreePaths={onAutoFillWorktreePaths}
                  onResetWorkerDefaults={onResetWorkerDefaults}
                  onResetWorkerOverrides={onResetWorkerOverrides}
                />
        ) : (
          <section className="card"><div className="small muted">Loading dashboard state...</div></section>
        )}
      </main>
    </div>
  );
}
