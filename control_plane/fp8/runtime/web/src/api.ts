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
} from './types';

async function parseJson<T>(response: Response): Promise<T> {
  const bodyText = await response.text();
  let data: (T & { error?: string; errors?: string[] }) | null = null;
  if (bodyText) {
    try {
      data = JSON.parse(bodyText) as T & { error?: string; errors?: string[] };
    } catch {
      const snippet = bodyText.slice(0, 160).replace(/\s+/g, ' ').trim();
      throw new Error(`request failed with status ${response.status}: expected JSON, received ${snippet || 'empty response'}`);
    }
  }
  if (!response.ok) {
    const errorText = data?.error || data?.errors?.join('\n') || `request failed with status ${response.status}`;
    throw new Error(errorText);
  }
  if (data === null) {
    throw new Error(`request failed with status ${response.status}: empty response body`);
  }
  return data;
}

async function postJson<T>(path: string, payload: Record<string, unknown>): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return parseJson<T>(response);
}

export async function fetchState(signal?: AbortSignal): Promise<DashboardState> {
  const response = await fetch('/api/state', { signal });
  return parseJson<DashboardState>(response);
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
