// csv-import.js — CSV file parsing and database import.

import { createReadStream } from 'fs';
import { insertRecord } from './db.js';

export class CSVImportError extends Error {
  constructor(message, row) {
    super(message);
    this.name = 'CSVImportError';
    this.row = row;
  }
}

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

export async function importCSV(filePath, options = {}) {
  const {
    fieldNames = [],
    hasHeader = true,
    tagsColumn = null,
    priceColumn = null,
  } = options;

  if (fieldNames.length === 0) {
    throw new Error('fieldNames must not be empty');
  }

  return new Promise((resolve, reject) => {
    const imported = [];
    const errors = [];
    let lineNumber = 0;
    let headers = null;

    const stream = createReadStream(filePath, { encoding: 'utf8' });
    const lines = [];
    let buffer = '';

    stream.on('data', (chunk) => {
      buffer += chunk;
      const splits = buffer.split('\n');
      buffer = splits.pop();
      lines.push(...splits);
    });

    stream.on('end', async () => {
      if (buffer) lines.push(buffer);

      try {
        for (let i = 0; i < lines.length; i++) {
          lineNumber = i + 1;
          const line = lines[i].trim();

          if (!line) continue;

          const values = parseCSVLine(line);

          if (hasHeader && !headers) {
            headers = values;
            continue;
          }

          const record = {};
          for (const field of fieldNames) {
            const colIndex = headers
              ? headers.indexOf(field)
              : fieldNames.indexOf(field);
            if (colIndex === -1 || !values[colIndex]) {
              throw new CSVImportError(
                `Field "${field}" not found or empty`,
                lineNumber
              );
            }
            const value = values[colIndex];
            record[field] = isNaN(value) ? value : parseFloat(value);
          }

          let tags = [];
          if (tagsColumn) {
            const tagColIndex = headers
              ? headers.indexOf(tagsColumn)
              : tagsColumn;
            if (tagColIndex !== -1 && values[tagColIndex]) {
              tags = values[tagColIndex]
                .split(';')
                .map((t) => t.trim())
                .filter((t) => t);
            }
          }

          let price = null;
          if (priceColumn) {
            const priceColIndex = headers
              ? headers.indexOf(priceColumn)
              : priceColumn;
            if (priceColIndex !== -1 && values[priceColIndex]) {
              price = parseFloat(values[priceColIndex]);
            }
          }

          const dbRecord = await insertRecord(record, tags, price);
          imported.push(dbRecord);
        }

        resolve({ imported, errors });
      } catch (error) {
        if (error instanceof CSVImportError) {
          errors.push(error);
          resolve({ imported, errors });
        } else {
          reject(error);
        }
      }
    });

    stream.on('error', reject);
  });
}
