import { IdbConext } from './IdbKit';
import { WaSqliteContext } from './waSqliteKit';

export class PersistenceSupportDb {
  private _idbConext: IdbConext;
  private _sqliteDb: WaSqliteContext;
  constructor(idbConext: IdbConext, sqliteDb: WaSqliteContext) {
    this._idbConext = idbConext;
    this._sqliteDb = sqliteDb;
  }

  async saveTableAsync(tableName: string) {
    if (this._idbConext.isTableExists(tableName)) {
      await this._idbConext.deleteTableAsync(tableName);
    }
    const dt = await this._sqliteDb.getDataTableAsync(tableName);
    await this._idbConext.createTableAsync(dt.tableName);
    await this._idbConext.saveTableAsync(dt, dt.tableName);
  }
} 
