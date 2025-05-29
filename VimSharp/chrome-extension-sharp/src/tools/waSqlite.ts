// https://github.com/rhashimoto/wa-sqlite
import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs'
import * as SQLite from 'wa-sqlite'
import Handlebars from 'handlebars'

// DataTable column interface
export interface DataTableColumn {
  name: string; // column name
  type: string; // column type, e.g. 'TEXT', 'INTEGER'
}

// DataTable interface
export interface DataTable {
  tableName: string;
  columns: DataTableColumn[];
  data: any[]; // array of row objects
}

/**
 * Create a table in SQLite database based on DataTable definition
 * @param dt - DataTable definition
 * @param tableName - Table name (override dt.tableName if provided)
 */
export async function createTableAsync(dt: DataTable, tableName?: string) {
  const name = tableName || dt.tableName;
  // Compose column definitions
  const columnsDef = dt.columns.map(col => `${col.name} ${col.type}`).join(', ');
  const sql = `CREATE TABLE IF NOT EXISTS ${name} (${columnsDef})`;
  await execSqliteAsync(sql);
}

export async function hello() {
  const module = await SQLiteESMFactory()
  const sqlite3 = SQLite.Factory(module)
  // 開啟資料庫（可指定 VFS，例如 'myDB', 'file:myDB?vfs=IDBBatchAtomicVFS' 等）
  const db = await sqlite3.open_v2('supportDb')
  // 執行 SQL
  await sqlite3.exec(db, `CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)`)
  await sqlite3.exec(db, `INSERT INTO test (name) VALUES ('Vue3')`)
  // 查詢資料
  await sqlite3.exec(db, `SELECT * FROM test`, (row, _columns) => {
    console.log(row)
  })
  // 關閉資料庫
  await sqlite3.close(db)
}

/**
 * Execute a SQL statement with Handlebars template and auto-close the database
 * @param sql - SQL statement (Handlebars template)
 * @param parameters - SQL parameters (object for template context)
 */
export async function execSqliteAsync(sql: string, parameters: any = {}) {
  const module = await SQLiteESMFactory();
  const sqlite3 = SQLite.Factory(module);
  const db = await sqlite3.open_v2('supportDb');
  // Use Handlebars to compile and render SQL
  const template = Handlebars.compile(sql);
  const lastSql = template(parameters);
  await sqlite3.exec(db, lastSql);
  await sqlite3.close(db);
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
  const valuesStr = columns.map(col => `{{${col}}}`).join(', ');
  const sql = `INSERT INTO ${name} (${columnsStr}) VALUES (${valuesStr})`;

  const module = await SQLiteESMFactory();
  const sqlite3 = SQLite.Factory(module);
  const db = await sqlite3.open_v2('supportDb');
  const template = Handlebars.compile(sql);
  for (const row of dt.data) {
      const lastSql = template(row);
      await sqlite3.exec(db, lastSql);
  }
  await sqlite3.close(db);
}
