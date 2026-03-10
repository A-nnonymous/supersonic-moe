import type {
  ConfigSaveResponse,
  DashboardState,
  LaunchResponse,
  StopAllResponse,
  StopWorkersResponse,
} from './types';

async function parseJson<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T & { error?: string; errors?: string[] };
  if (!response.ok) {
    const errorText = data.error || data.errors?.join('\n') || `request failed with status ${response.status}`;
    throw new Error(errorText);
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

export function saveConfig(configText: string): Promise<ConfigSaveResponse> {
  return postJson<ConfigSaveResponse>('/api/config', { config_text: configText });
}

export function launchWorkers(restart: boolean): Promise<LaunchResponse> {
  return postJson<LaunchResponse>('/api/launch', { restart });
}

export function stopWorkers(): Promise<StopWorkersResponse> {
  return postJson<StopWorkersResponse>('/api/stop', {});
}

export function stopAll(): Promise<StopAllResponse> {
  return postJson<StopAllResponse>('/api/stop-all', {});
}
