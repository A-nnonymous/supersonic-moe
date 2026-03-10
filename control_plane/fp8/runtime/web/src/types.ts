export type TabKey = 'overview' | 'operations' | 'settings';

export type CommandMap = {
  serve: string;
  up: string;
};

export type DashboardMode = {
  state: string;
  cold_start: boolean;
  reason: string;
  config_path: string;
  persist_config_path: string;
};

export type ProviderQueueItem = {
  resource_pool: string;
  provider: string;
  model: string;
  priority: number;
  binary: string;
  binary_found: boolean;
  api_key_present: boolean;
  connection_quality: number;
  work_quality: number;
  score: number;
  latency_ms: number | null;
  active_workers: number;
  last_failure: string;
};

export type MergeQueueItem = {
  agent: string;
  branch: string;
  submit_strategy: string;
  merge_target: string;
  worker_identity: string;
  manager_identity: string;
  status: string;
  manager_action: string;
};

export type RuntimeWorker = {
  agent: string;
  resource_pool: string;
  provider: string;
  model: string;
  branch: string;
  status: string;
};

export type HeartbeatAgent = {
  agent: string;
  role: string;
  state: string;
  last_seen: string;
  evidence: string;
  expected_next_checkin: string;
  escalation: string;
};

export type ProcessSnapshot = {
  resource_pool: string;
  provider: string;
  model: string;
  pid: number;
  alive: boolean;
  returncode: number | null;
  worktree_path: string;
  log_path: string;
  command: string[];
};

export type BacklogItem = {
  id: string;
  owner: string;
  status: string;
  gate: string;
  title: string;
};

export type GateItem = {
  id: string;
  name: string;
  status: string;
  owner: string;
};

export type ConfigProject = {
  repository_name?: string;
  local_repo_root?: string;
  paddle_repo_path?: string;
  integration_branch?: string;
  base_branch?: string;
  manager_git_identity?: {
    name?: string;
    email?: string;
  };
  dashboard?: {
    host?: string;
    port?: number;
  };
};

export type ConfigWorker = {
  agent: string;
  task_id?: string;
  resource_pool?: string;
  resource_pool_queue?: string[];
  branch?: string;
  git_identity?: {
    name?: string;
    email?: string;
  };
  submit_strategy?: string;
  test_command?: string;
};

export type ConfigShape = {
  project?: ConfigProject;
  resource_pools?: Record<string, {
    priority?: number;
    provider?: string;
    model?: string;
  }>;
  workers?: ConfigWorker[];
};

export type DashboardState = {
  updated_at: string;
  last_event: string;
  mode: DashboardMode;
  project: ConfigProject;
  commands: CommandMap;
  manager_report: string;
  runtime: { workers?: RuntimeWorker[] };
  heartbeats: { agents?: HeartbeatAgent[] };
  backlog: { items?: BacklogItem[] };
  gates: { gates?: GateItem[] };
  processes: Record<string, ProcessSnapshot>;
  provider_queue: ProviderQueueItem[];
  merge_queue: MergeQueueItem[];
  config: ConfigShape;
  config_text: string;
  validation_errors: string[];
  launch_blockers: string[];
};

export type ConfigSaveResponse = {
  ok: boolean;
  validation_errors: string[];
  launch_blockers: string[];
  cold_start: boolean;
};

export type LaunchResponse = {
  ok: boolean;
  launched?: Array<Record<string, unknown>>;
  failures?: Array<{ agent: string; error: string }>;
  errors?: string[];
};

export type StopWorkersResponse = {
  ok: boolean;
  stopped: string[];
};

export type StopAllResponse = {
  ok: boolean;
  stop_agents: boolean;
  listener_port?: number;
  listener_released?: boolean;
  stopped_workers?: string[];
  warning?: string;
};
