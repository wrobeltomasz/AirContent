// admin.js — Admin panel routes for configuring the system.

import express from 'express';
import { getConfig, saveConfig, resetConfig } from './config-store.js';
import * as db from './db.js';
import { resetPool } from './db.js';
import { validateRecord } from './validation.js';

const router = express.Router();

router.get('/api/config', (req, res) => {
  res.json(getConfig());
});

router.post('/api/config', express.json(), async (req, res) => {
  try {
    const newConfig = req.body;
    saveConfig(newConfig);
    if (newConfig.database) await resetPool();
    res.json({ success: true, config: getConfig() });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

router.post('/api/config/reset', (req, res) => {
  resetConfig();
  res.json({ success: true, config: getConfig() });
});

router.get('/api/records', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 1000);
    const offset = parseInt(req.query.offset) || 0;
    const tag = req.query.tag || null;

    const records = await db.getRecords(limit, offset, tag);
    const count = await db.countRecords();

    res.json({ records, count, limit, offset });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/api/records/:id', async (req, res) => {
  try {
    const record = await db.getRecordById(parseInt(req.params.id));
    if (!record) {
      res.status(404).json({ error: 'Record not found' });
      return;
    }
    res.json(record);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/api/records', express.json(), async (req, res) => {
  try {
    const { metadata, tags = [] } = req.body;

    if (!metadata) {
      res.status(400).json({ error: 'metadata is required' });
      return;
    }

    validateRecord(metadata);

    const record = await db.insertRecord(metadata, tags);

    const cfg = getConfig();
    cfg.metrics.business.recordsIngested =
      (cfg.metrics.business.recordsIngested || 0) + 1;
    try {
      saveConfig(cfg);
    } catch {
      /* non-fatal */
    }

    res.status(201).json(record);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

router.get('/api/db/test', async (req, res) => {
  await resetPool();
  try {
    await db.getPool().query('SELECT 1');
    res.json({ success: true, message: 'Connection successful' });
  } catch (error) {
    await resetPool();
    res.status(500).json({ success: false, message: error.message });
  }
});

router.delete('/api/records/:id', async (req, res) => {
  try {
    await db.deleteRecord(parseInt(req.params.id));
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post(
  '/api/csv/import',
  express.raw({ type: 'application/octet-stream' }),
  async (req, res) => {
    try {
      const { importCSV } = await import('./csv-import.js');
      const fs = await import('fs');
      const path = await import('path');
      const { fileURLToPath } = await import('url');
      const __dirname = path.dirname(fileURLToPath(import.meta.url));
      const tempPath = path.join(__dirname, `temp-${Date.now()}.csv`);

      fs.writeFileSync(tempPath, req.body);

      const options = {
        fieldNames: req.query.fields ? req.query.fields.split(',') : [],
        hasHeader: req.query.hasHeader !== 'false',
        tagsColumn: req.query.tagsColumn || null,
        priceColumn: req.query.priceColumn || null,
      };

      let result;
      try {
        result = await importCSV(tempPath, options);
      } finally {
        // Always remove the temp file, even if the import throws.
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }

      const cfg = getConfig();
      cfg.metrics.business.recordsIngested =
        (cfg.metrics.business.recordsIngested || 0) + result.imported.length;
      try {
        saveConfig(cfg);
      } catch {
        /* non-fatal */
      }

      res.json(result);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }
);

export default router;
