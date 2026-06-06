// config-store.js — runtime config management.
//
// Layers runtime overrides from runtime-config.json (admin panel) and
// environment variables on top of the defaults in config.js.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { config as defaultConfig, assertConfigConsistent } from './config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RUNTIME_CONFIG_PATH = path.join(__dirname, 'runtime-config.json');

let runtimeConfig = null;

function loadRuntimeConfig() {
  if (fs.existsSync(RUNTIME_CONFIG_PATH)) {
    try {
      return JSON.parse(fs.readFileSync(RUNTIME_CONFIG_PATH, 'utf8'));
    } catch {
      console.warn(`Failed to parse ${RUNTIME_CONFIG_PATH}, using defaults`);
    }
  }
  return {};
}

function envOverrides() {
  const env = process.env;
  return {
    server: {
      port: env.PORT ? parseInt(env.PORT, 10) : undefined,
    },
    database: {
      host: env.PG_HOST,
      port: env.PG_PORT ? parseInt(env.PG_PORT, 10) : undefined,
      database: env.PG_DATABASE,
      user: env.PG_USER,
      password: env.PG_PASSWORD,
      ssl: env.PG_SSL ? env.PG_SSL === 'true' : undefined,
    },
  };
}

function merge(base, override) {
  const result = JSON.parse(JSON.stringify(base));
  for (const key of Object.keys(override)) {
    if (override[key] === undefined) continue;
    if (
      typeof override[key] === 'object' &&
      override[key] !== null &&
      !Array.isArray(override[key])
    ) {
      result[key] = merge(result[key] || {}, override[key]);
    } else {
      result[key] = override[key];
    }
  }
  return result;
}

export function getConfig() {
  if (!runtimeConfig) {
    const runtime = loadRuntimeConfig();
    const env = envOverrides();
    runtimeConfig = merge(merge(defaultConfig, runtime), env);
    assertConfigConsistent(runtimeConfig);
  }
  return runtimeConfig;
}

export function saveConfig(newConfig) {
  assertConfigConsistent(newConfig);
  fs.writeFileSync(
    RUNTIME_CONFIG_PATH,
    JSON.stringify(newConfig, null, 2),
    'utf8'
  );
  runtimeConfig = newConfig;
}

export function resetConfig() {
  runtimeConfig = null;
  if (fs.existsSync(RUNTIME_CONFIG_PATH)) {
    fs.unlinkSync(RUNTIME_CONFIG_PATH);
  }
}
