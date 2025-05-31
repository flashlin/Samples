// https://github.com/rhashimoto/wa-sqlite
import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs'
import * as SQLite from 'wa-sqlite'
import Handlebars from 'handlebars'
import { DataTable, guessType } from './dataTypes'

/**
 * Create a table in SQLite database based on DataTable definition
 * @param dt - DataTable definition
 * @param tableName - Table name (override dt.tableName if provided)
 */
export async function createTableAsync(dt: DataTable, tableName?: string) {
  const name = tableName || dt.tableName;
  // Compose column definitions
  const columnsDef = dt.columns.map(col => `${col.name} ${col.type}`).join(', ');
  const sql = `CREATE TABLE ${name} (${columnsDef})`;
  await execSqliteAsync(sql);
}

export async function hello() {
  await withSQLiteDbAsync(async (sqlite3, db) => {
    await sqlite3.exec(db, `CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)`)
    await sqlite3.exec(db, `INSERT INTO test (name) VALUES ('Vue3'), ('React'), ('Angular')`)
    await sqlite3.exec(db, `SELECT * FROM test`, (row, _columns) => {
      console.log(row)
    })
  });
}

/**
 * Execute a SQL statement with Handlebars template and auto-close the database
 * @param sql - SQL statement (Handlebars template)
 * @param parameters - SQL parameters (object for template context)
 */
export async function execSqliteAsync(sql: string, parameters: any = {}) {
  const template = Handlebars.compile(sql);
  const lastSql = template(parameters);
  await withSQLiteDbAsync(async (sqlite3, db) => {
    await sqlite3.exec(db, lastSql);
  });
}

export async function querySqliteAsync(sql: string, parameters: any = {}): Promise<DataTable> {
  const template = Handlebars.compile(sql);
  const lastSql = template(parameters);
  const data: any[] = [];
  let columns: string[] = [];
  let columnTypes: string[] = [];
  await withSQLiteDbAsync(async (sqlite3, db) => {
    await sqlite3.exec(db, lastSql, (row, cols) => {
      if (columns.length === 0) {
        columns = cols;
        // 只在第一筆資料時推斷型別
        columnTypes = row.map((v: any) => guessType(v));
        console.log("querySqliteAsync", lastSql)
      }
      const obj: any = {};
      cols.forEach((col, index) => {
        obj[col] = row[index];
      });
      data.push(obj);
    });
  });
  console.log("querySqliteAsync result", data)
  // 組成 DataTable
  return {
    tableName: '', // 查詢無明確表名
    columns: columns.map((name, idx) => ({ name, type: columnTypes[idx] || 'TEXT' })),
    data
  };
}

/**
 * Drop a table if it exists
 * @param tableName - Table name to drop
 */
export async function dropTableAsync(tableName: string) {
  const sql = `DROP TABLE IF EXISTS ${tableName}`;
  await execSqliteAsync(sql);
}

/**
 * Insert data into a table using DataTable definition
 * @param dt - DataTable (需有 data 欄位: 陣列，每個元素為物件)
 * @param tableName - Table name
 */
export async function insertDataTableAsync(dt: DataTable, tableName: string) {
  const name = tableName || dt.tableName;
  const columns = dt.columns.map(col => col.name);
  const columnsStr = columns.join(', ');
  const valuesStr = dt.columns
    .map(col => (col.type === 'TEXT' || col.type === 'DATE' ? `'{{${col.name}}}'` : `{{${col.name}}}`))
    .join(', ');
  const sql = `INSERT INTO ${name} (${columnsStr}) VALUES (${valuesStr})`;
  const template = Handlebars.compile(sql);
  await withSQLiteDbAsync(async (sqlite3, db) => {
    for (const row of dt.data) {
      const lastSql = template(row);
      await sqlite3.exec(db, lastSql);
    }
  });
}

/**
 * Utility function to handle SQLite connection lifecycle
 * @param callback - async function that receives (sqlite3, db)
 */
const module = await SQLiteESMFactory();
export async function withSQLiteDbAsync(callback: (sqlite3: SQLiteAPI, db: number) => Promise<void>) {
  const sqlite3 = SQLite.Factory(module);
  const db = await sqlite3.open_v2('supportDb');
  try {
    await callback(sqlite3, db);
  } finally {
    await sqlite3.close(db);
  }
}
