// car-lifecycle.js
// Full pipeline:
//   1. Connect to PostgreSQL, create `cars` and `road_segments` tables
//   2. Insert 60+ realistic European/Polish-market car records
//   3. Insert 16 road-segment records (Polish voivodeships)
//   4. Train TensorFlow.js regression model on 4 000 synthetic samples
//   5. Predict remaining useful life (years) for every car
//   6. Update predictions in DB
//   7. Export results/car-lifecycle-results.json + .csv
//
// Usage:  node car-lifecycle.js
// DB:     host=localhost port=5432 dbname=postgres user=postgres password=1234

import pg from 'pg';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { Pool } = pg;
const CURRENT_YEAR = 2026;

// ─────────────────────────────────────────────────────────────────────────────
// Database connection
// ─────────────────────────────────────────────────────────────────────────────
const pool = new Pool({
  host: 'localhost',
  port: 5432,
  database: 'postgres',
  user: 'postgres',
  password: '1234',
});

// ─────────────────────────────────────────────────────────────────────────────
// Polish road-segment data (16 voivodeships, source: GDDKiA / BDL GUS 2024)
// road_quality: 1 (bad) – 10 (excellent highways)
// pct_highway, pct_urban, pct_rural: usage mix typical for each region
// ─────────────────────────────────────────────────────────────────────────────
const ROAD_SEGMENTS = [
  { region: 'Masovian',        road_quality: 8, pct_highway: 35, pct_urban: 45, pct_rural: 20 },
  { region: 'Lesser Poland',   road_quality: 7, pct_highway: 30, pct_urban: 40, pct_rural: 30 },
  { region: 'Silesian',        road_quality: 8, pct_highway: 40, pct_urban: 45, pct_rural: 15 },
  { region: 'Greater Poland',  road_quality: 7, pct_highway: 32, pct_urban: 35, pct_rural: 33 },
  { region: 'Lower Silesian',  road_quality: 7, pct_highway: 33, pct_urban: 38, pct_rural: 29 },
  { region: 'Lodz',            road_quality: 6, pct_highway: 28, pct_urban: 40, pct_rural: 32 },
  { region: 'Pomeranian',      road_quality: 7, pct_highway: 30, pct_urban: 42, pct_rural: 28 },
  { region: 'Kuyavian',        road_quality: 6, pct_highway: 25, pct_urban: 38, pct_rural: 37 },
  { region: 'Subcarpathian',   road_quality: 5, pct_highway: 15, pct_urban: 30, pct_rural: 55 },
  { region: 'Lublin',          road_quality: 5, pct_highway: 14, pct_urban: 28, pct_rural: 58 },
  { region: 'Warmian-Masurian',road_quality: 5, pct_highway: 12, pct_urban: 25, pct_rural: 63 },
  { region: 'West Pomeranian', road_quality: 6, pct_highway: 22, pct_urban: 35, pct_rural: 43 },
  { region: 'Opole',           road_quality: 7, pct_highway: 30, pct_urban: 33, pct_rural: 37 },
  { region: 'Holy Cross',      road_quality: 5, pct_highway: 12, pct_urban: 28, pct_rural: 60 },
  { region: 'Lubusz',          road_quality: 6, pct_highway: 24, pct_urban: 30, pct_rural: 46 },
  { region: 'Podlaskie',       road_quality: 4, pct_highway: 10, pct_urban: 22, pct_rural: 68 },
];

// ─────────────────────────────────────────────────────────────────────────────
// Car sample data — 62 realistic European/Polish-market records
// maintenance_score: 1-10  (10 = dealer-serviced, on time, every time)
// road_quality:      1-10  (matches ROAD_SEGMENTS entries)
// highway_pct:       0-100 (% of total km on motorways/expressways)
// ─────────────────────────────────────────────────────────────────────────────
const CAR_DATA = [
  // Toyota — known for reliability
  { make:'Toyota',  model:'Corolla',   year:2010, mileage_km:185000, maintenance:8, road_quality:7, accidents:1, highway_pct:30, fuel:'gasoline', engine_cc:1600 },
  { make:'Toyota',  model:'Corolla',   year:2015, mileage_km:95000,  maintenance:9, road_quality:8, accidents:0, highway_pct:40, fuel:'gasoline', engine_cc:1600 },
  { make:'Toyota',  model:'Yaris',     year:2019, mileage_km:42000,  maintenance:9, road_quality:9, accidents:0, highway_pct:20, fuel:'hybrid',   engine_cc:1500 },
  { make:'Toyota',  model:'Avensis',   year:2008, mileage_km:245000, maintenance:7, road_quality:5, accidents:2, highway_pct:55, fuel:'diesel',   engine_cc:2000 },
  { make:'Toyota',  model:'RAV4',      year:2020, mileage_km:58000,  maintenance:9, road_quality:8, accidents:0, highway_pct:45, fuel:'hybrid',   engine_cc:2500 },

  // Volkswagen
  { make:'VW',      model:'Golf',      year:2012, mileage_km:168000, maintenance:7, road_quality:7, accidents:1, highway_pct:35, fuel:'diesel',   engine_cc:2000 },
  { make:'VW',      model:'Golf',      year:2018, mileage_km:85000,  maintenance:8, road_quality:8, accidents:0, highway_pct:40, fuel:'gasoline', engine_cc:1400 },
  { make:'VW',      model:'Passat',    year:2013, mileage_km:198000, maintenance:6, road_quality:7, accidents:1, highway_pct:60, fuel:'diesel',   engine_cc:2000 },
  { make:'VW',      model:'Polo',      year:2021, mileage_km:24000,  maintenance:9, road_quality:9, accidents:0, highway_pct:25, fuel:'gasoline', engine_cc:1000 },
  { make:'VW',      model:'Tiguan',    year:2017, mileage_km:108000, maintenance:8, road_quality:7, accidents:0, highway_pct:38, fuel:'diesel',   engine_cc:2000 },

  // Skoda
  { make:'Skoda',   model:'Octavia',   year:2014, mileage_km:148000, maintenance:8, road_quality:7, accidents:0, highway_pct:42, fuel:'diesel',   engine_cc:2000 },
  { make:'Skoda',   model:'Fabia',     year:2020, mileage_km:38000,  maintenance:9, road_quality:9, accidents:0, highway_pct:18, fuel:'gasoline', engine_cc:1000 },
  { make:'Skoda',   model:'Superb',    year:2016, mileage_km:138000, maintenance:8, road_quality:8, accidents:1, highway_pct:55, fuel:'diesel',   engine_cc:2000 },
  { make:'Skoda',   model:'Karoq',     year:2021, mileage_km:31000,  maintenance:9, road_quality:8, accidents:0, highway_pct:35, fuel:'gasoline', engine_cc:1500 },

  // Ford
  { make:'Ford',    model:'Focus',     year:2011, mileage_km:178000, maintenance:6, road_quality:5, accidents:2, highway_pct:28, fuel:'gasoline', engine_cc:1600 },
  { make:'Ford',    model:'Focus',     year:2018, mileage_km:72000,  maintenance:8, road_quality:8, accidents:0, highway_pct:40, fuel:'diesel',   engine_cc:2000 },
  { make:'Ford',    model:'Fiesta',    year:2016, mileage_km:88000,  maintenance:7, road_quality:7, accidents:1, highway_pct:22, fuel:'gasoline', engine_cc:1000 },
  { make:'Ford',    model:'Mondeo',    year:2009, mileage_km:223000, maintenance:5, road_quality:4, accidents:3, highway_pct:50, fuel:'diesel',   engine_cc:2000 },
  { make:'Ford',    model:'Kuga',      year:2018, mileage_km:95000,  maintenance:8, road_quality:8, accidents:0, highway_pct:45, fuel:'diesel',   engine_cc:2000 },

  // Opel
  { make:'Opel',    model:'Astra',     year:2012, mileage_km:158000, maintenance:6, road_quality:6, accidents:1, highway_pct:30, fuel:'diesel',   engine_cc:1700 },
  { make:'Opel',    model:'Astra',     year:2018, mileage_km:82000,  maintenance:7, road_quality:8, accidents:0, highway_pct:35, fuel:'gasoline', engine_cc:1400 },
  { make:'Opel',    model:'Corsa',     year:2020, mileage_km:35000,  maintenance:8, road_quality:8, accidents:0, highway_pct:20, fuel:'gasoline', engine_cc:1200 },
  { make:'Opel',    model:'Insignia',  year:2015, mileage_km:165000, maintenance:6, road_quality:7, accidents:1, highway_pct:60, fuel:'diesel',   engine_cc:2000 },

  // Renault
  { make:'Renault', model:'Clio',      year:2017, mileage_km:75000,  maintenance:7, road_quality:7, accidents:0, highway_pct:25, fuel:'gasoline', engine_cc:900  },
  { make:'Renault', model:'Megane',    year:2013, mileage_km:145000, maintenance:6, road_quality:5, accidents:2, highway_pct:30, fuel:'diesel',   engine_cc:1500 },
  { make:'Renault', model:'Kadjar',    year:2019, mileage_km:62000,  maintenance:8, road_quality:8, accidents:0, highway_pct:38, fuel:'diesel',   engine_cc:1500 },
  { make:'Renault', model:'Duster',    year:2016, mileage_km:118000, maintenance:7, road_quality:4, accidents:0, highway_pct:10, fuel:'diesel',   engine_cc:1500 },

  // BMW
  { make:'BMW',     model:'3 Series',  year:2014, mileage_km:155000, maintenance:7, road_quality:9, accidents:1, highway_pct:55, fuel:'diesel',   engine_cc:2000 },
  { make:'BMW',     model:'5 Series',  year:2017, mileage_km:115000, maintenance:8, road_quality:9, accidents:0, highway_pct:65, fuel:'diesel',   engine_cc:3000 },
  { make:'BMW',     model:'1 Series',  year:2019, mileage_km:62000,  maintenance:8, road_quality:8, accidents:0, highway_pct:40, fuel:'gasoline', engine_cc:1500 },
  { make:'BMW',     model:'X5',        year:2016, mileage_km:128000, maintenance:8, road_quality:8, accidents:1, highway_pct:50, fuel:'diesel',   engine_cc:3000 },

  // Audi
  { make:'Audi',    model:'A4',        year:2013, mileage_km:192000, maintenance:7, road_quality:8, accidents:1, highway_pct:58, fuel:'diesel',   engine_cc:2000 },
  { make:'Audi',    model:'A3',        year:2018, mileage_km:78000,  maintenance:8, road_quality:9, accidents:0, highway_pct:42, fuel:'gasoline', engine_cc:1400 },
  { make:'Audi',    model:'Q5',        year:2017, mileage_km:105000, maintenance:8, road_quality:8, accidents:0, highway_pct:48, fuel:'diesel',   engine_cc:2000 },
  { make:'Audi',    model:'A6',        year:2012, mileage_km:218000, maintenance:6, road_quality:8, accidents:2, highway_pct:62, fuel:'diesel',   engine_cc:2000 },

  // Fiat
  { make:'Fiat',    model:'500',       year:2019, mileage_km:45000,  maintenance:7, road_quality:8, accidents:0, highway_pct:15, fuel:'gasoline', engine_cc:900  },
  { make:'Fiat',    model:'Punto',     year:2010, mileage_km:142000, maintenance:5, road_quality:4, accidents:2, highway_pct:18, fuel:'gasoline', engine_cc:1200 },
  { make:'Fiat',    model:'Tipo',      year:2018, mileage_km:72000,  maintenance:7, road_quality:6, accidents:0, highway_pct:28, fuel:'diesel',   engine_cc:1600 },

  // Peugeot
  { make:'Peugeot', model:'308',       year:2016, mileage_km:98000,  maintenance:7, road_quality:7, accidents:1, highway_pct:32, fuel:'diesel',   engine_cc:1500 },
  { make:'Peugeot', model:'208',       year:2020, mileage_km:38000,  maintenance:8, road_quality:8, accidents:0, highway_pct:20, fuel:'gasoline', engine_cc:1200 },
  { make:'Peugeot', model:'3008',      year:2019, mileage_km:68000,  maintenance:8, road_quality:8, accidents:0, highway_pct:38, fuel:'diesel',   engine_cc:1500 },
  { make:'Peugeot', model:'508',       year:2013, mileage_km:182000, maintenance:6, road_quality:7, accidents:1, highway_pct:55, fuel:'diesel',   engine_cc:2000 },

  // Hyundai
  { make:'Hyundai', model:'i30',       year:2016, mileage_km:88000,  maintenance:8, road_quality:8, accidents:0, highway_pct:35, fuel:'diesel',   engine_cc:1600 },
  { make:'Hyundai', model:'Tucson',    year:2019, mileage_km:62000,  maintenance:9, road_quality:8, accidents:0, highway_pct:42, fuel:'diesel',   engine_cc:2000 },
  { make:'Hyundai', model:'i20',       year:2021, mileage_km:25000,  maintenance:9, road_quality:9, accidents:0, highway_pct:22, fuel:'gasoline', engine_cc:1000 },

  // Kia
  { make:'Kia',     model:'Ceed',      year:2017, mileage_km:92000,  maintenance:8, road_quality:8, accidents:0, highway_pct:38, fuel:'diesel',   engine_cc:1600 },
  { make:'Kia',     model:'Sportage',  year:2020, mileage_km:45000,  maintenance:9, road_quality:8, accidents:0, highway_pct:40, fuel:'diesel',   engine_cc:1600 },
  { make:'Kia',     model:'Picanto',   year:2022, mileage_km:15000,  maintenance:10,road_quality:9, accidents:0, highway_pct:15, fuel:'gasoline', engine_cc:1000 },

  // Nissan
  { make:'Nissan',  model:'Qashqai',   year:2017, mileage_km:102000, maintenance:8, road_quality:7, accidents:0, highway_pct:40, fuel:'diesel',   engine_cc:1500 },
  { make:'Nissan',  model:'Leaf',      year:2020, mileage_km:52000,  maintenance:9, road_quality:9, accidents:0, highway_pct:30, fuel:'electric', engine_cc:0    },
  { make:'Nissan',  model:'Juke',      year:2018, mileage_km:75000,  maintenance:7, road_quality:7, accidents:1, highway_pct:28, fuel:'gasoline', engine_cc:1000 },

  // Mercedes
  { make:'Mercedes',model:'C-Class',   year:2015, mileage_km:158000, maintenance:7, road_quality:9, accidents:0, highway_pct:60, fuel:'diesel',   engine_cc:2100 },
  { make:'Mercedes',model:'A-Class',   year:2019, mileage_km:65000,  maintenance:8, road_quality:9, accidents:0, highway_pct:42, fuel:'gasoline', engine_cc:1600 },
  { make:'Mercedes',model:'GLC',       year:2018, mileage_km:85000,  maintenance:9, road_quality:9, accidents:0, highway_pct:50, fuel:'diesel',   engine_cc:2200 },

  // Seat
  { make:'Seat',    model:'Leon',      year:2016, mileage_km:98000,  maintenance:7, road_quality:7, accidents:1, highway_pct:35, fuel:'diesel',   engine_cc:1600 },
  { make:'Seat',    model:'Ibiza',     year:2020, mileage_km:35000,  maintenance:8, road_quality:8, accidents:0, highway_pct:22, fuel:'gasoline', engine_cc:1000 },
  { make:'Seat',    model:'Ateca',     year:2019, mileage_km:68000,  maintenance:8, road_quality:7, accidents:0, highway_pct:38, fuel:'diesel',   engine_cc:2000 },

  // Mazda
  { make:'Mazda',   model:'CX-5',      year:2018, mileage_km:82000,  maintenance:9, road_quality:8, accidents:0, highway_pct:42, fuel:'diesel',   engine_cc:2200 },
  { make:'Mazda',   model:'3',         year:2021, mileage_km:28000,  maintenance:9, road_quality:9, accidents:0, highway_pct:35, fuel:'gasoline', engine_cc:2000 },

  // Volvo
  { make:'Volvo',   model:'V60',       year:2017, mileage_km:122000, maintenance:8, road_quality:9, accidents:0, highway_pct:58, fuel:'diesel',   engine_cc:2000 },
  { make:'Volvo',   model:'XC60',      year:2020, mileage_km:55000,  maintenance:9, road_quality:9, accidents:0, highway_pct:52, fuel:'diesel',   engine_cc:2000 },

  // High-mileage stress cases
  { make:'VW',      model:'Transporter',year:2010,mileage_km:312000, maintenance:6, road_quality:5, accidents:1, highway_pct:65, fuel:'diesel',   engine_cc:2000 },
  { make:'Toyota',  model:'HiLux',     year:2012, mileage_km:289000, maintenance:7, road_quality:3, accidents:1, highway_pct:20, fuel:'diesel',   engine_cc:2500 },

  // Near end-of-life
  { make:'Opel',    model:'Vectra',    year:2003, mileage_km:285000, maintenance:4, road_quality:3, accidents:4, highway_pct:25, fuel:'gasoline', engine_cc:1800 },
  { make:'Fiat',    model:'Seicento',  year:2002, mileage_km:198000, maintenance:3, road_quality:3, accidents:3, highway_pct:10, fuel:'gasoline', engine_cc:900  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Physics-inspired life-cycle formula (generates ground truth for training)
// Returns remaining useful life in years [0, 15].
// ─────────────────────────────────────────────────────────────────────────────
function formulaLifecycle(car) {
  const age   = CURRENT_YEAR - car.year;
  const base  = 15;
  const agePen      = age * 0.38;
  const mileagePen  = (car.mileage_km / 300000) * 8.5;
  const maintEffect = (car.maintenance - 5) * 0.65;
  const roadPen     = ((10 - car.road_quality) / 9) * 2.8;
  const accPen      = car.accidents * 1.6;
  const hwBonus     = (car.highway_pct / 100) * 1.2; // highway easier on drivetrain
  return Math.max(0, Math.min(15, base - agePen - mileagePen + maintEffect - roadPen - accPen + hwBonus));
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature vector: 6 normalized inputs → [0, 1]
// ─────────────────────────────────────────────────────────────────────────────
function toFeatureVector(car) {
  const age = CURRENT_YEAR - car.year;
  return [
    Math.min(age, 30) / 30,               // age_years
    Math.min(car.mileage_km, 350000) / 350000, // mileage
    car.maintenance / 10,                  // maintenance quality
    car.road_quality / 10,                 // road quality
    Math.min(car.accidents, 6) / 6,        // accident count
    (car.highway_pct || 30) / 100,         // highway fraction
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Generate 4 000 synthetic training samples
// ─────────────────────────────────────────────────────────────────────────────
function generateTrainingData(n = 4000) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const age       = Math.random() * 28;
    const mileage   = Math.min(age * 14000 * (0.4 + Math.random() * 1.2), 350000);
    const maintenance = 1 + Math.random() * 9;
    const road_quality = 1 + Math.random() * 9;
    const accidents = Math.floor(Math.random() * 7);
    const highway_pct = Math.floor(Math.random() * 100);
    const synth = { year: CURRENT_YEAR - age, mileage_km: mileage, maintenance, road_quality, accidents, highway_pct };
    xs.push(toFeatureVector(synth));
    const label = formulaLifecycle(synth) / 15; // normalise to [0,1]
    ys.push([label]);
  }
  return { xs, ys };
}

// ─────────────────────────────────────────────────────────────────────────────
// Build + train TF regression model
// ─────────────────────────────────────────────────────────────────────────────
async function trainModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [6], units: 64, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.15 }),
      tf.layers.dense({ units: 32, activation: 'relu' }),
      tf.layers.dense({ units: 16, activation: 'relu' }),
      tf.layers.dense({ units: 1,  activation: 'sigmoid' }),
    ],
  });
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError', metrics: ['mae'] });

  const { xs, ys } = generateTrainingData(4000);
  const xT = tf.tensor2d(xs);
  const yT = tf.tensor2d(ys);

  process.stdout.write('   Training');
  await model.fit(xT, yT, {
    epochs: 120,
    batchSize: 64,
    validationSplit: 0.1,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 24 === 0) process.stdout.write('.');
      },
    },
  });
  console.log(' done');

  xT.dispose();
  yT.dispose();
  return model;
}

async function mlPredict(model, car) {
  const inp = tf.tensor2d([toFeatureVector(car)]);
  const out = model.predict(inp);
  const raw = (await out.data())[0];
  inp.dispose();
  out.dispose();
  return Math.max(0, Math.min(15, raw * 15));
}

// ─────────────────────────────────────────────────────────────────────────────
// Database helpers
// ─────────────────────────────────────────────────────────────────────────────
async function setupSchema(client) {
  // road_segments
  await client.query(`
    CREATE TABLE IF NOT EXISTS road_segments (
      id SERIAL PRIMARY KEY,
      region VARCHAR(60) NOT NULL,
      road_quality INTEGER NOT NULL,
      pct_highway INTEGER NOT NULL,
      pct_urban   INTEGER NOT NULL,
      pct_rural   INTEGER NOT NULL,
      created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  // cars
  await client.query(`
    CREATE TABLE IF NOT EXISTS cars (
      id SERIAL PRIMARY KEY,
      make          VARCHAR(30) NOT NULL,
      model         VARCHAR(30) NOT NULL,
      year          INTEGER NOT NULL,
      mileage_km    INTEGER NOT NULL,
      maintenance_score FLOAT NOT NULL,
      road_quality  FLOAT NOT NULL,
      accident_count INTEGER NOT NULL DEFAULT 0,
      highway_pct   INTEGER NOT NULL DEFAULT 30,
      fuel_type     VARCHAR(20),
      engine_cc     INTEGER,
      age_years     INTEGER,
      ml_remaining_life_yrs   FLOAT,
      formula_remaining_yrs   FLOAT,
      lifecycle_category      VARCHAR(20),
      created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  await client.query(`CREATE INDEX IF NOT EXISTS cars_make_idx ON cars (make)`);
  await client.query(`CREATE INDEX IF NOT EXISTS cars_life_idx ON cars (ml_remaining_life_yrs DESC NULLS LAST)`);
}

async function clearTables(client) {
  await client.query('DELETE FROM cars');
  await client.query('DELETE FROM road_segments');
}

async function insertRoadSegments(client) {
  for (const rs of ROAD_SEGMENTS) {
    await client.query(
      `INSERT INTO road_segments (region, road_quality, pct_highway, pct_urban, pct_rural)
       VALUES ($1,$2,$3,$4,$5)`,
      [rs.region, rs.road_quality, rs.pct_highway, rs.pct_urban, rs.pct_rural]
    );
  }
}

async function insertCar(client, car) {
  const age = CURRENT_YEAR - car.year;
  const result = await client.query(
    `INSERT INTO cars
       (make, model, year, mileage_km, maintenance_score, road_quality, accident_count, highway_pct, fuel_type, engine_cc, age_years)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
     RETURNING id`,
    [car.make, car.model, car.year, car.mileage_km, car.maintenance, car.road_quality,
     car.accidents, car.highway_pct ?? 30, car.fuel, car.engine_cc, age]
  );
  return result.rows[0].id;
}

async function updateCarPrediction(client, id, mlLife, formulaLife, category) {
  await client.query(
    `UPDATE cars SET ml_remaining_life_yrs=$1, formula_remaining_yrs=$2, lifecycle_category=$3 WHERE id=$4`,
    [mlLife, formulaLife, category, id]
  );
}

function lifecycleCategory(years) {
  if (years < 1)  return 'END-OF-LIFE';
  if (years < 3)  return 'CRITICAL';
  if (years < 6)  return 'AGING';
  if (years < 10) return 'GOOD';
  return 'EXCELLENT';
}

// ─────────────────────────────────────────────────────────────────────────────
// Export helpers
// ─────────────────────────────────────────────────────────────────────────────
function toCSV(rows) {
  const cols = [
    'id','make','model','year','mileage_km','age_years',
    'maintenance_score','road_quality','accident_count','highway_pct',
    'fuel_type','engine_cc',
    'ml_remaining_life_yrs','formula_remaining_yrs','lifecycle_category',
  ];
  const header = cols.join(',');
  const lines = rows.map(r =>
    cols.map(c => {
      const v = r[c] ?? '';
      return typeof v === 'string' && (v.includes(',') || v.includes('"'))
        ? `"${v.replace(/"/g, '""')}"` : v;
    }).join(',')
  );
  return [header, ...lines].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n══════════════════════════════════════════════════════════════');
  console.log('  AirContent — Car & Road Lifecycle Estimation Pipeline');
  console.log('══════════════════════════════════════════════════════════════\n');

  const client = await pool.connect();

  try {
    // 1. Schema
    console.log('1. Creating / verifying database schema...');
    await setupSchema(client);
    await clearTables(client);
    console.log('   ✓ Tables: cars, road_segments\n');

    // 2. Road segments
    console.log('2. Inserting road-segment data (16 Polish voivodeships)...');
    await insertRoadSegments(client);
    console.log(`   ✓ ${ROAD_SEGMENTS.length} road segments\n`);

    // 3. Cars
    console.log(`3. Inserting ${CAR_DATA.length} car records...`);
    for (const car of CAR_DATA) {
      car._id = await insertCar(client, car);
    }
    console.log(`   ✓ ${CAR_DATA.length} cars\n`);

    // 4. Train
    console.log('4. Training TensorFlow.js model (4 000 samples, 120 epochs)...');
    const model = await trainModel();
    console.log('   ✓ Model ready\n');

    // 5. Predict
    console.log('5. Predicting remaining useful life for each car...');
    const results = [];

    for (const car of CAR_DATA) {
      const age = CURRENT_YEAR - car.year;
      const mlLife      = Number((await mlPredict(model, car)).toFixed(2));
      const formulaLife = Number(formulaLifecycle(car).toFixed(2));
      const category    = lifecycleCategory(mlLife);

      await updateCarPrediction(client, car._id, mlLife, formulaLife, category);

      results.push({
        id: car._id,
        make: car.make,
        model: car.model,
        year: car.year,
        mileage_km: car.mileage_km,
        age_years: age,
        maintenance_score: car.maintenance,
        road_quality: car.road_quality,
        accident_count: car.accidents,
        highway_pct: car.highway_pct ?? 30,
        fuel_type: car.fuel,
        engine_cc: car.engine_cc,
        ml_remaining_life_yrs: mlLife,
        formula_remaining_yrs: formulaLife,
        lifecycle_category: category,
      });
    }

    model.dispose();
    results.sort((a, b) => b.ml_remaining_life_yrs - a.ml_remaining_life_yrs);
    console.log(`   ✓ ${results.length} predictions done\n`);

    // 6. Console table
    console.log('6. Results:\n');
    const H = ['Make & Model', 'Year', 'Mileage', 'Age', 'ML Life', 'Formula', 'Category'];
    console.log(
      H[0].padEnd(22) + H[1].padEnd(6) + H[2].padEnd(11) +
      H[3].padEnd(5)  + H[4].padEnd(9)  + H[5].padEnd(9) + H[6]
    );
    console.log('─'.repeat(76));
    for (const r of results) {
      console.log(
        `${r.make} ${r.model}`.padEnd(22) +
        String(r.year).padEnd(6) +
        `${r.mileage_km.toLocaleString()}km`.padEnd(11) +
        `${r.age_years}y`.padEnd(5) +
        `${r.ml_remaining_life_yrs}yr`.padEnd(9) +
        `${r.formula_remaining_yrs}yr`.padEnd(9) +
        r.lifecycle_category
      );
    }

    // 7. Stats
    const avg = results.reduce((s, r) => s + r.ml_remaining_life_yrs, 0) / results.length;
    const cats = {};
    results.forEach(r => { cats[r.lifecycle_category] = (cats[r.lifecycle_category] || 0) + 1; });

    console.log('\n' + '─'.repeat(76));
    console.log(`Average remaining life: ${avg.toFixed(2)} years`);
    console.log('By category:', Object.entries(cats).sort().map(([k,v]) => `${k}:${v}`).join('  '));

    // Road segment summary
    console.log('\n7. Road segment summary:\n');
    console.log('Region'.padEnd(22) + 'Quality'.padEnd(10) + 'Highway%'.padEnd(11) + 'Urban%'.padEnd(9) + 'Rural%');
    console.log('─'.repeat(60));
    for (const rs of ROAD_SEGMENTS.sort((a,b) => b.road_quality - a.road_quality)) {
      console.log(
        rs.region.padEnd(22) +
        String(rs.road_quality).padEnd(10) +
        `${rs.pct_highway}%`.padEnd(11) +
        `${rs.pct_urban}%`.padEnd(9) +
        `${rs.pct_rural}%`
      );
    }

    // 8. Save files
    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

    const jsonPath = path.join(resultsDir, 'car-lifecycle-results.json');
    const csvPath  = path.join(resultsDir, 'car-lifecycle-results.csv');
    const roadPath = path.join(resultsDir, 'road-segments.json');

    const summary = {
      generated_at: new Date().toISOString(),
      model_type: 'TensorFlow.js Sequential (6 features, 4 layers)',
      training_samples: 4000,
      cars_analyzed: results.length,
      avg_remaining_life_years: Number(avg.toFixed(2)),
      categories: cats,
      results,
    };

    fs.writeFileSync(jsonPath, JSON.stringify(summary, null, 2), 'utf8');
    fs.writeFileSync(csvPath,  toCSV(results), 'utf8');
    fs.writeFileSync(roadPath, JSON.stringify({ road_segments: ROAD_SEGMENTS }, null, 2), 'utf8');

    console.log('\n8. Files saved:');
    console.log(`   ${jsonPath}`);
    console.log(`   ${csvPath}`);
    console.log(`   ${roadPath}`);
    console.log('\n══════════════════════════════════════════════════════════════\n');

  } finally {
    client.release();
    await pool.end();
  }
}

main().catch(err => {
  console.error('\nFatal error:', err.message);
  process.exit(1);
});
