// https://github.com/rhashimoto/wa-sqlite
import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs'
import * as SQLite from 'wa-sqlite'
import Handlebars from 'handlebars'
import { DataTable, DataTableColumn, guessType } from './dataTypes'

export class WaSqliteContext 
{
  _dbName: string = 'supportDb';
  _module: any = null;

  async openAsync(dbName: string) {
    this._dbName = dbName;
    this._module = await SQLiteESMFactory();
  }

  /**
   * Execute a SQL statement with Handlebars template and auto-close the database
   * @param sql - SQL statement (Handlebars template)
   * @param parameters - SQL parameters (object for template context)
   */
  async execAsync(sql: string, parameters: any = {}) {
    const template = Handlebars.compile(sql);
    const lastSql = template(parameters);
    await this.withSQLiteDbAsync(async (sqlite3, db) => {
      await sqlite3.exec(db, lastSql);
    });
  }


  /**
   * 執行 SQL 查詢並回傳 DataTable 結果
   * @param sql - SQL 查詢語句（可使用 Handlebars 模板語法）
   * @param parameters - 傳入模板的參數物件（可選，預設為空物件）
   * @returns Promise<DataTable> 查詢結果，包含欄位資訊與資料陣列
   */
  async queryAsync(sql: string, parameters: any = {}): Promise<DataTable> {
    const template = Handlebars.compile(sql);
    const lastSql = template(parameters);
    const data: any[] = [];
    let columns: string[] = [];
    let columnTypes: string[] = [];
    await this.withSQLiteDbAsync(async (sqlite3, db) => {
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
    // 組成 DataTable
    return {
      tableName: '', // 查詢無明確表名
      columns: columns.map((name, idx) => ({ name, type: columnTypes[idx] || 'TEXT' })),
      data
    };
  }

  /**
   * Create a table in SQLite database based on DataTable definition
   * @param dt - DataTable definition
   * @param tableName - Table name (override dt.tableName if provided)
   */
  async createTableAsync(dt: DataTable, tableName?: string) {
    const name = tableName || dt.tableName;
    // Compose column definitions
    const columnsDef = dt.columns.map(col => `${col.name} ${col.type}`).join(', ');
    const sql = `CREATE TABLE ${name} (${columnsDef})`;
    await this.execAsync(sql);
  }

  /**
   * Drop a table if it exists
   * @param tableName - Table name to drop
   */
  async dropTableAsync(tableName: string) {
    const sql = `DROP TABLE IF EXISTS ${tableName}`;
    await this.execAsync(sql);
  }

  /**
   * Insert data into a table using DataTable definition
   * @param dt - DataTable (需有 data 欄位: 陣列，每個元素為物件)
   * @param tableName - Table name
   */
  async insertDataTableAsync(dt: DataTable, tableName: string) {
    const name = tableName || dt.tableName;
    const columns = dt.columns.map(col => col.name);
    const columnsStr = columns.join(', ');
    const valuesStr = dt.columns
      .map(col => (col.type === 'TEXT' || col.type === 'DATE' ? `'{{${col.name}}}'` : `{{${col.name}}}`))
      .join(', ');
    const sql = `INSERT INTO ${name} (${columnsStr}) VALUES (${valuesStr})`;
    const template = Handlebars.compile(sql);
    await this.withSQLiteDbAsync(async (sqlite3, db) => {
      for (const row of dt.data) {
        const lastSql = template(row);
        //console.info("waSqliteKit::insertDataTableAsync", lastSql);
        await sqlite3.exec(db, lastSql);
      }
    });
  }

  async getTableSchemaAsync(tableName: string): Promise<DataTableColumn[]> {
    const columns: DataTableColumn[] = [];
    await this.withSQLiteDbAsync(async (sqlite3, db) => {
      await sqlite3.exec(db, `PRAGMA table_info(${tableName});`, (row: any, cols: string[]) => {
        const obj: any = {};
        cols.forEach((col, idx) => {
          obj[col] = row[idx];
        });
        columns.push({
          name: obj.name,
          type: obj.type,
        });
      });
    });
    return columns;
  }

  async getDataTableAsync(tableName: string): Promise<DataTable> {
    // 取得欄位資訊
    const columns = await this.getTableSchemaAsync(tableName);
    // 查詢所有資料
    const data: any[] = [];
    await this.withSQLiteDbAsync(async (sqlite3, db) => {
      await sqlite3.exec(db, `SELECT * FROM ${tableName}`, (row: any, cols: string[]) => {
        const obj: any = {};
        cols.forEach((col, idx) => {
          obj[col] = row[idx];
        });
        data.push(obj);
      });
    });
    // 組成 DataTable
    return {
      tableName,
      columns,
      data
    };
  }

  /**
   * 取得所有資料表名稱
   * @returns Promise<string[]> 資料表名稱陣列
   */
  async getAllTableNamesAsync(): Promise<string[]> {
    const sql = `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`;
    const result = await this.queryAsync(sql);
    return result.data.map(row => row.name);
  }

  async withSQLiteDbAsync(callback: (sqlite3: SQLiteAPI, db: number) => Promise<void>) {
    const sqlite3 = SQLite.Factory(this._module);
    const db = await sqlite3.open_v2(this._dbName, 
      SQLite.SQLITE_OPEN_CREATE | SQLite.SQLITE_OPEN_READWRITE);
    try {
      await callback(sqlite3, db);
    } finally {
      await sqlite3.close(db);
    }
  }

  /**
   * 檢查指定的資料表是否存在
   * @param tableName 資料表名稱
   * @returns Promise<boolean> 是否存在
   */
  async isTableExistsAsync(tableName: string): Promise<boolean> {
    const sql = `SELECT name FROM sqlite_master WHERE type='table' AND name='{{tableName}}'`;
    const result = await this.queryAsync(sql, { tableName });
    return result.data.length > 0;
  }
}
