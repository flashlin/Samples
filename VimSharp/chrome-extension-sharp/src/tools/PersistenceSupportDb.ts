import { DataTableColumn } from './dataTypes';
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
      await this._idbConext.dropTableAsync(tableName);
    }
    const dt = await this._sqliteDb.getDataTableAsync(tableName);
    await this._idbConext.createTableAsync(dt.tableName);
    await this._idbConext.saveTableAsync(dt, dt.tableName);
  }

  async saveTableSchemaAsync(tableName: string, columns: DataTableColumn[]) {
    const tableSchemaName = 'tableSchemas';
    if (!this._idbConext.isTableExists(tableSchemaName)) {
      await this._idbConext.createTableAsync(tableSchemaName, "name");
    }
    const tableSchema = {
      name: tableName,
      columns: columns
    };
    await this._idbConext.upsertRowAsync(tableSchemaName, tableSchema);
  }

  async restoreAllTablesAsync() {
    const tableNames = this._idbConext.getAllTableNames();
    for (const tableName of tableNames) {
      const dt = await this._idbConext.getTableAsync(tableName);
      console.info("PersistenceSupportDb::restoreAllTablesAsync", dt);
      await this._sqliteDb.dropTableAsync(tableName);
      await this._sqliteDb.insertDataTableAsync(dt, tableName);
    }
  }
} 
