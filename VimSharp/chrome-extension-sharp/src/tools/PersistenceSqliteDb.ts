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

  async saveTableSchemas() {
    // 取得所有 table 名稱與 schema
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
    // 儲存到 idb
    const idb = await openDB<SchemaDB>(this.dbName, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('tableSchemas')) {
          db.createObjectStore('tableSchemas', { keyPath: 'name' });
        }
      },
    });
    const tx = idb.transaction('tableSchemas', 'readwrite');
    const store = tx.objectStore('tableSchemas');
    for (const schema of tableSchemas) {
      await store.put(schema);
    }
    await tx.done;
  }
} 