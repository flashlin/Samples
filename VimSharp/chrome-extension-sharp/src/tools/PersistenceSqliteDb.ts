import { openDB, DBSchema } from 'idb';

// 提取 withSQLiteDbAsync 型別
export type WithSQLiteDbAsyncFn = (callback: (sqlite3: any, db: any) => Promise<void>) => Promise<void>;

interface TableSchema {
  name: string;
  schema: string;
}

interface SchemaDB extends DBSchema {
  tableSchemas: {
    key: string;
    value: TableSchema;
  };
}

export class PersistenceSqliteDb {
  private dbName = 'sqlite-schema-db';
  private withSQLiteDbAsync: WithSQLiteDbAsyncFn;

  constructor(dbName: string, withSQLiteDbAsync: WithSQLiteDbAsyncFn) {
    this.dbName = dbName;
    this.withSQLiteDbAsync = withSQLiteDbAsync;
  }

  /**
   * 取得所有 sqlite table 的 schema
   */
  async getSqliteTableSchemasAsync(): Promise<TableSchema[]> {
    const tableSchemas: TableSchema[] = [];
    await this.withSQLiteDbAsync(async (sqlite3: any, db: any) => {
      // 查詢所有 user table 名稱
      const tables: string[] = [];
      await sqlite3.exec(db, `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`, (row: any) => {
        if (row[0] && typeof row[0] === 'string') {
          tables.push(row[0]);
        }
      });
      // 查詢每個 table 的 schema
      for (const tableName of tables) {
        let schema = '';
        await sqlite3.exec(db, `SELECT sql FROM sqlite_master WHERE type='table' AND name='${tableName}'`, (row: any) => {
          if (row[0] && typeof row[0] === 'string') {
            schema = row[0];
          }
        });
        tableSchemas.push({ name: tableName, schema });
      }
    });
    return tableSchemas;
  }

  async saveTableSchemasWithDataAsync() {
    // 取得所有 table 名稱與 schema
    const tableSchemas = await this.getSqliteTableSchemasAsync();
    for (const schema of tableSchemas) {
      await this.saveTableSchemaAsync(schema);
      await this.saveTableDataAsync(schema.name);
    }
  }

  /**
   * 儲存單一 table schema 到 idb
   */
  async saveTableSchemaAsync(tableSchema: TableSchema) {
    // 先開啟 DB，檢查 object store 是否存在
    let idb: any;
    idb = await openDB<SchemaDB>(this.dbName); // 不指定 version
    if (!idb.objectStoreNames.contains('tableSchemas')) {
      const newVersion = idb.version + 1;
      idb.close();
      idb = await openDB<SchemaDB>(this.dbName, newVersion, {
        upgrade(db) {
          if (!db.objectStoreNames.contains('tableSchemas')) {
            db.createObjectStore('tableSchemas', { keyPath: 'name' });
          }
        },
      });
    }
    const tx = idb.transaction('tableSchemas', 'readwrite');
    const store = tx.objectStore('tableSchemas');
    await store.put(tableSchema);
    await tx.done;
  }

  /**
   * 將某一個 sqlite table 的所有資料儲存到 idb
   * @param tableName - 要儲存的 table 名稱
   */
  async saveTableDataAsync(tableName: string) {
    // 取得所有資料
    let rows: any[] = [];
    await this.withSQLiteDbAsync(async (sqlite3: any, db: any) => {
      await sqlite3.exec(db, `SELECT * FROM ${tableName}`, (row: any, columns: string[]) => {
        const obj: any = {};
        columns.forEach((col, idx) => {
          obj[col] = row[idx];
        });
        rows.push(obj);
      });
    });
    // 儲存到 idb（每個 table 對應一個 object store，keyPath 設為 'id'）
    let idb = await openDB(this.dbName); // 不指定 version
    if (!idb.objectStoreNames.contains(tableName)) {
      idb = await this.createIdbTable(idb, tableName);
    } else {
      // 如果已存在，先清空資料
      await this.clearIdbTableData(idb, tableName);
    }
    const tx = idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    for (const row of rows) {
      await store.put(row);
    }
    await tx.done;
  }

  /**
   * 清空指定 object store 的所有資料
   */
  private async clearIdbTableData(idb: any, tableName: string) {
    if (idb.objectStoreNames.contains(tableName)) {
      const clearTx = idb.transaction(tableName, 'readwrite');
      await clearTx.objectStore(tableName).clear();
      await clearTx.done;
    }
  }

  /**
   * 建立指定名稱的 object store（table），若尚未存在
   * @param tableName
   * @param idb 已開啟的資料庫實例
   * @returns 升級後的資料庫實例
   */
  private async createIdbTable(idb: any, tableName: string) {
    const newVersion = idb.version + 1;
    idb.close();
    return await openDB(this.dbName, newVersion, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(tableName)) {
          db.createObjectStore(tableName, { autoIncrement: true });
        }
      },
    });
  }

  /**
   * 從 idb 取得所有 table schema 與資料，並還原到 sqlite
   */
  async restoreTableSchemasWithDataAsync() {
    // 1. 取得所有 table schema
    const idb = await openDB<any>(this.dbName);
    if (!idb.objectStoreNames.contains('tableSchemas')) {
      return;
    }
    const allSchemas = await idb.getAll('tableSchemas');
    // 2. 建立所有 sqlite table schema 並還原資料
    await this.withSQLiteDbAsync(async (sqlite3: any, db: any) => {
      for (const schema of allSchemas) {
        // 建立 table schema
        if (schema.schema) {
          await sqlite3.exec(db, schema.schema);
        }
        // 取得該 table 的所有資料
        if (idb.objectStoreNames.contains(schema.name as string)) {
          const tx = idb.transaction(schema.name as string, 'readonly');
          const store = tx.objectStore(schema.name as string);
          const allRows: any[] = await store.getAll();
          await tx.done;
          // 還原資料
          for (const row of allRows) {
            const columns = Object.keys(row);
            const placeholders = columns.map(_col => '?').join(',');
            const sql = `INSERT OR REPLACE INTO ${schema.name} (${columns.join(',')}) VALUES (${placeholders})`;
            await sqlite3.run(db, sql, columns.map(col => row[col]));
          }
        }
      }
    });
  }
} 