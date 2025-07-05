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

  constructor(withSQLiteDbAsync: WithSQLiteDbAsyncFn) {
    this.withSQLiteDbAsync = withSQLiteDbAsync;
  }

  /**
   * 取得所有 sqlite table 的 schema
   */
  async getSqliteTableSchemas(): Promise<TableSchema[]> {
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

  async saveTableSchemas() {
    // 取得所有 table 名稱與 schema
    const tableSchemas = await this.getSqliteTableSchemas();
    for (const schema of tableSchemas) {
      await this.saveTableSchema(schema);
    }
  }

  /**
   * 儲存單一 table schema 到 idb
   */
  async saveTableSchema(tableSchema: TableSchema) {
    const idb = await openDB<SchemaDB>(this.dbName, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('tableSchemas')) {
          db.createObjectStore('tableSchemas', { keyPath: 'name' });
        }
      },
    });
    const tx = idb.transaction('tableSchemas', 'readwrite');
    const store = tx.objectStore('tableSchemas');
    await store.put(tableSchema);
    await tx.done;
  }

  /**
   * 將某一個 sqlite table 的所有資料儲存到 idb
   * @param tableName - 要儲存的 table 名稱
   */
  async saveTableData(tableName: string) {
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
    const idb = await openDB(this.dbName, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(tableName)) {
          db.createObjectStore(tableName, { keyPath: 'id' });
        }
      },
    });
    const tx = idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    for (const row of rows) {
      await store.put(row);
    }
    await tx.done;
  }
} 