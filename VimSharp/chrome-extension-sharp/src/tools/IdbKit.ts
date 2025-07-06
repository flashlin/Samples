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
    //await this.deleteTableAsync(tableName);
    const tx = this._idb.transaction(tableName, 'readwrite');
    const store = tx.objectStore(tableName);
    for (const row of dt.data) {
      await store.add(row);
    }
    await tx.done;
  }
  async getTableAsync(tableName: string): Promise<DataTable> {
    if( this._idb == null) {
      throw new Error('idb not open');
    }
    const tx = this._idb.transaction(tableName, 'readonly');
    const store = tx.objectStore(tableName);
    const rows = await store.getAll();
    return {
      tableName: tableName,
      columns: [],
      data: rows,
    };
  }
  isTableExists(tableName: string): boolean {
    if (this._idb == null) {
      throw new Error('idb not open');
    }
    return this._idb.objectStoreNames.contains(tableName);
  }
}
