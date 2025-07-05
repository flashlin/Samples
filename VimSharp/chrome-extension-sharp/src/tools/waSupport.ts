// https://github.com/rhashimoto/wa-sqlite
import { DataTable } from './dataTypes'
import { WaSqliteContext } from './waSqliteKit';
import { PersistenceSqliteDb } from './PersistenceSqliteDb';

const waSqlite = new WaSqliteContext();
await waSqlite.openAsync('supportDb');

/**
 * Execute a SQL statement with Handlebars template and auto-close the database
 * @param sql - SQL statement (Handlebars template)
 * @param parameters - SQL parameters (object for template context)
 */
export async function execSqliteAsync(sql: string, parameters: any = {}) {
  await waSqlite.execAsync(sql, parameters);
}

export async function querySqliteAsync(sql: string, parameters: any = {}): Promise<DataTable> {
  return await waSqlite.queryAsync(sql, parameters);
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
  const sql = `CREATE TABLE ${name} (${columnsDef})`;
  await execSqliteAsync(sql);
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
  await waSqlite.insertDataTableAsync(dt, tableName);
}


// const idb = new PersistenceSqliteDb('supportDb', withSQLiteDbAsync);
// export async function backupSqliteDbAsync() {
//   await idb.saveTableSchemasWithDataAsync();
// }
// export async function restoreSqliteDbAsync() {
//   await idb.restoreTableSchemasWithDataAsync();
// }
