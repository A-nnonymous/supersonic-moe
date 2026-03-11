import type {
  ConfigSection,
  ConfigShape,
  ConfigSaveResponse,
  ConfigValidationResponse,
  DashboardState,
  LaunchStrategy,
  LaunchResponse,
  SilentModeResponse,
  StopAllResponse,
  StopWorkersResponse,
  ValidationIssue,
} from './types';

type ErrorPayload = {
  error?: string;
  errors?: string[];
  failures?: Array<{ agent?: string; error?: string }>;
  validation_issues?: ValidationIssue[];
  validation_errors?: string[];
  launch_blockers?: string[];
};

function summarizeItems(items: string[], limit = 4): string {
  if (!items.length) {
    return '';
  }
  if (items.length <= limit) {
    return items.join('; ');
  }
  return `${items.slice(0, limit).join('; ')}; ... (+${items.length - limit} more)`;
}

function extractErrorText(data: ErrorPayload | null, requestPath: string, status: number): string {
  if (!data) {
    return `request ${requestPath} failed with status ${status}`;
  }
  if (data.error) {
    return data.error;
  }
  if (data.failures?.length) {
    const failures = data.failures.map((item) => `${item.agent || 'worker'}: ${item.error || 'unknown launch failure'}`);
    return `request ${requestPath} failed with status ${status}: ${summarizeItems(failures)}`;
  }
  if (data.validation_issues?.length) {
    const issues = data.validation_issues.map((item) => `${item.field}: ${item.message}`);
    return `request ${requestPath} failed with status ${status}: ${summarizeItems(issues)}`;
  }
  if (data.launch_blockers?.length) {
    return `request ${requestPath} failed with status ${status}: ${summarizeItems(data.launch_blockers)}`;
  }
  if (data.validation_errors?.length) {
    return `request ${requestPath} failed with status ${status}: ${summarizeItems(data.validation_errors)}`;
  }
  if (data.errors?.length) {
    return `request ${requestPath} failed with status ${status}: ${summarizeItems(data.errors)}`;
  }
  return `request ${requestPath} failed with status ${status}`;
}

async function parseJson<T>(response: Response, requestPath: string): Promise<T> {
  const bodyText = await response.text();
  let data: (T & ErrorPayload) | null = null;
  if (bodyText) {
    try {
      data = JSON.parse(bodyText) as T & ErrorPayload;
    } catch {
      const snippet = bodyText.slice(0, 160).replace(/\s+/g, ' ').trim();
      throw new Error(`request ${requestPath} failed with status ${response.status}: expected JSON, received ${snippet || 'empty response'}`);
    }
  }
  if (!response.ok) {
    throw new Error(extractErrorText(data, requestPath, response.status));
  }
  if (data === null) {
    throw new Error(`request ${requestPath} failed with status ${response.status}: empty response body`);
  }
  return data;
}

async function postJson<T>(path: string, payload: Record<string, unknown>): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return parseJson<T>(response, path);
}

export async function fetchState(signal?: AbortSignal): Promise<DashboardState> {
  const path = '/api/state';
  const response = await fetch(path, { signal });
  return parseJson<DashboardState>(response, path);
}

export function validateConfig(config: ConfigShape): Promise<ConfigValidationResponse> {
  return postJson<ConfigValidationResponse>('/api/config/validate', { config });
}

export function saveConfig(config: ConfigShape): Promise<ConfigSaveResponse> {
  return postJson<ConfigSaveResponse>('/api/config', { config });
}

export function validateConfigSection(section: ConfigSection, value: unknown): Promise<ConfigValidationResponse> {
  return postJson<ConfigValidationResponse>('/api/config/validate-section', { section, value });
}

export function saveConfigSection(section: ConfigSection, value: unknown): Promise<ConfigSaveResponse> {
  return postJson<ConfigSaveResponse>('/api/config/section', { section, value });
}

export function launchWorkers(
  restart: boolean,
  launchPolicy?: { strategy: LaunchStrategy; provider?: string; model?: string },
): Promise<LaunchResponse> {
  return postJson<LaunchResponse>('/api/launch', { restart, ...launchPolicy });
}

export function stopWorkers(): Promise<StopWorkersResponse> {
  return postJson<StopWorkersResponse>('/api/stop', {});
}

export function stopAll(): Promise<StopAllResponse> {
  return postJson<StopAllResponse>('/api/stop-all', {});
}

export function enableSilentMode(): Promise<SilentModeResponse> {
  return postJson<SilentModeResponse>('/api/silent', {});
}
