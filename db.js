// db.js — PostgreSQL database layer.

import pg from 'pg';
import { getConfig } from './config-store.js';

const { Pool } = pg;
let pool = null;

export function getPool() {
  if (!pool) {
    const cfg = getConfig().database;
    pool = new Pool({
      host: cfg.host,
      port: cfg.port,
      database: cfg.database,
      user: cfg.user,
      password: cfg.password || undefined,
      ssl: cfg.ssl || false,
    });
  }
  return pool;
}

export async function initSchema() {
  const client = await getPool().connect();
  try {
    const table = getConfig().database.table;
    await client.query(`
      CREATE TABLE IF NOT EXISTS ${table} (
        id SERIAL PRIMARY KEY,
        metadata JSONB NOT NULL,
        tags TEXT[] DEFAULT ARRAY[]::TEXT[],
        price FLOAT8,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);
    await client.query(`
      CREATE INDEX IF NOT EXISTS ${table}_tags_idx
      ON ${table} USING GIN(tags);
    `);
    await client.query(`
      CREATE INDEX IF NOT EXISTS ${table}_created_idx
      ON ${table} (created_at DESC);
    `);
  } finally {
    client.release();
  }
}

export async function insertRecord(metadata, tags = [], price = null) {
  const cfg = getConfig();
  const table = cfg.database.table;
  const result = await getPool().query(
    `INSERT INTO ${table} (metadata, tags, price)
     VALUES ($1, $2, $3)
     RETURNING id, metadata, tags, price, created_at`,
    [JSON.stringify(metadata), tags, price]
  );
  return result.rows[0];
}

export async function getRecords(limit = 100, offset = 0, tag = null) {
  const cfg = getConfig();
  const table = cfg.database.table;
  let query = `SELECT * FROM ${table}`;
  const params = [];

  if (tag) {
    query += ` WHERE $1 = ANY(tags)`;
    params.push(tag);
  }

  query += ` ORDER BY created_at DESC LIMIT $${params.length + 1} OFFSET $${
    params.length + 2
  }`;
  params.push(limit, offset);

  const result = await getPool().query(query, params);
  return result.rows;
}

export async function getRecordById(id) {
  const cfg = getConfig();
  const table = cfg.database.table;
  const result = await getPool().query(
    `SELECT * FROM ${table} WHERE id = $1`,
    [id]
  );
  return result.rows[0] || null;
}

export async function deleteRecord(id) {
  const cfg = getConfig();
  const table = cfg.database.table;
  await getPool().query(`DELETE FROM ${table} WHERE id = $1`, [id]);
}

export async function countRecords() {
  const cfg = getConfig();
  const table = cfg.database.table;
  const result = await getPool().query(`SELECT COUNT(*) FROM ${table}`);
  return parseInt(result.rows[0].count, 10);
}

export async function closePool() {
  if (pool) {
    await pool.end();
    pool = null;
  }
}

export async function resetPool() {
  if (pool) {
    try { await pool.end(); } catch { /* ignore */ }
    pool = null;
  }
}
