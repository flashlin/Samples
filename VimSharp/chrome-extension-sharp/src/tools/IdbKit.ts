import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { DataTable } from './dataTypes';

export class IdbConext {
  private _idb: IDBPDatabase<unknown> | null = null;
  private _dbName: string = '';
  async openAsync(dbName: string) {
    console.info('IdbConext::openAsync', dbName);
    this._dbName = dbName;
    this._idb = await openDB(this._dbName);
  }
  async closeAsync() {
    if (this._idb) {
      await this._idb.close();
      this._idb = null;
    }
  }
  async createTableAsync(tableName: string, keyField: string='_id') {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    const newVersion = this._idb.version + 1;
    this._idb.close();
    this._idb = await openDB(this._dbName, newVersion, {
      upgrade(db) {
        db.createObjectStore(tableName, { keyPath: keyField, autoIncrement: true });
      },
    });
  }
  async deleteTableAsync(tableName: string) {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readwrite');
    await tx.objectStore(tableName).clear();
    await tx.done;
  }
  async saveTableAsync(dt: DataTable, tableName: string) {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    console.info("IdbConext::saveTableAsync", dt, tableName);
    const tx = this._idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    for (const row of dt.data) {
      await store.add(row);
    }
    await tx.done;
  }
  async addRowAsync(tableName: string, row: any) {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    await store.add(row);
    await tx.done;
  }
  async getRowsAsync(tableName: string): Promise<any[]> {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readonly');
    const store = tx.objectStore(tableName);
    const rows = await store.getAll();
    return rows;
  }
  async upsertRowAsync(tableName: string, row: any) {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    // 取得 keyPath
    const keyPath = store.keyPath as string;
    if (!keyPath) {
      throw new Error('table has no keyPath');
    }
    const keyValue = row[keyPath];
    let exists = false;
    if (keyValue !== undefined) {
      const existing = await store.get(keyValue);
      exists = !!existing;
    }
    if (exists) {
      await store.put(row); // 更新
    } else {
      await store.add(row); // 新增
    }
    await tx.done;
  }
  async deleteRowAsync(tableName: string, key: string) {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    await store.delete(key);
    await tx.done;
  }
  async dropTableAsync(tableName: string) {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    // 刪除 objectStore 需要升級資料庫版本
    const newVersion = this._idb.version + 1;
    this._idb.close();
    this._idb = await openDB(this._dbName, newVersion, {
      upgrade(db) {
        if (db.objectStoreNames.contains(tableName)) {
          db.deleteObjectStore(tableName);
        }
      },
    });
  }
  isTableExists(tableName: string): boolean {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    return this._idb.objectStoreNames.contains(tableName);
  }
  getAllTableNames(): string[] {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    // objectStoreNames 是一個 DOMStringList，要轉成陣列
    return Array.from(this._idb.objectStoreNames);
  }
}
