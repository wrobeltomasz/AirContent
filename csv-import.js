// csv-import.js — CSV file parsing and database import.

import { createReadStream } from 'fs';
import { insertRecord } from './db.js';
import { validateRecord, ValidationError } from './validation.js';

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
          const lineNumber = i + 1;
          const line = lines[i].trim();

          if (!line) continue;

          const values = parseCSVLine(line);

          if (hasHeader && !headers) {
            headers = values;
            continue;
          }

          try {
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
              const raw = values[colIndex];
              const num = parseFloat(raw);
              record[field] = isNaN(num) ? raw : num;
            }

            validateRecord(record);

            let tags = [];
            if (tagsColumn) {
              const tagColIndex = headers
                ? headers.indexOf(tagsColumn)
                : parseInt(tagsColumn, 10);
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
                : parseInt(priceColumn, 10);
              if (priceColIndex !== -1 && values[priceColIndex]) {
                const parsed = parseFloat(values[priceColIndex]);
                if (!isNaN(parsed)) price = parsed;
              }
            }

            const dbRecord = await insertRecord(record, tags, price);
            imported.push(dbRecord);
          } catch (rowError) {
            if (
              rowError instanceof CSVImportError ||
              rowError instanceof ValidationError
            ) {
              errors.push({ row: lineNumber, message: rowError.message });
            } else {
              throw rowError;
            }
          }
        }

        resolve({ imported, errors });
      } catch (error) {
        reject(error);
      }
    });

    stream.on('error', reject);
  });
}
