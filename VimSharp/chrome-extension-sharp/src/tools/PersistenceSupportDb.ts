import { DataTable, DataTableColumn } from './dataTypes';
import { IdbConext } from './IdbKit';
import { WaSqliteContext } from './waSqliteKit';

const tableSchemaName = 'tableSchemas';

interface TableSchema {
  name: string;
  columns: DataTableColumn[];
}

export class PersistenceSupportDb {
  private _idbConext: IdbConext;
  private _sqliteDb: WaSqliteContext;

  constructor(idbConext: IdbConext, sqliteDb: WaSqliteContext) {
    this._idbConext = idbConext;
    this._sqliteDb = sqliteDb;
  }

  async saveTableAsync(dt: DataTable) {
    const tableName = dt.tableName;
    if (this._idbConext.isTableExists(tableName)) {
      await this._idbConext.dropTableAsync(tableName);
    }
    await this.saveTableSchemaAsync(tableName, dt.columns);
    await this._idbConext.createTableAsync(dt.tableName);
    await this._idbConext.saveTableAsync(dt, dt.tableName);
  }

  async saveTableSchemaAsync(tableName: string, columns: DataTableColumn[]) {
    if (!this._idbConext.isTableExists(tableSchemaName)) {
      await this._idbConext.createTableAsync(tableSchemaName, "name");
    }
    const tableSchema = {
      name: tableName,
      columns: columns
    };
    await this._idbConext.upsertRowAsync(tableSchemaName, tableSchema);
  }

  async syncFromSqliteTablesAsync() {
    const tableNames = await this._sqliteDb.getAllTableNamesAsync();
    const tableSchemaList = await this.getAllTableSchemasAsync();
    const deletedTableSchemaList = tableSchemaList.filter(schema => 
      !tableNames.includes(schema.name)
    );
    for(const deletedTableSchema of deletedTableSchemaList) {
      await this.deleteTableSchemaAsync(deletedTableSchema.name);
      await this._idbConext.dropTableAsync(deletedTableSchema.name);
    }
    const newTableNameList = tableNames.filter(tableName => 
      !tableSchemaList.some(schema => schema.name === tableName)
    );
    for(const newTableName of newTableNameList) {
      const dt = await this._sqliteDb.getDataTableAsync(newTableName);
      await this.saveTableAsync(dt);
    }
  }

  async restoreAllTablesAsync() {
    const tableSchemaList = await this.getAllTableSchemasAsync();
    for(const tableSchema of tableSchemaList) {
      const tableName = tableSchema.name;
      const columns = tableSchema.columns;
      const rows = await this._idbConext.getRowsAsync(tableName);
      const dataTable = {
        tableName,
        columns,
        data: rows
      }
      if( await this._sqliteDb.isTableExistsAsync(tableName)) {
        await this._sqliteDb.dropTableAsync(tableName);
      }
      await this._sqliteDb.createTableAsync(dataTable);
      this._sqliteDb.insertDataTableAsync(dataTable, tableName);
    }
  }

  async getAllTableSchemasAsync(): Promise<TableSchema[]> {
    const tableSchemaList = await this._idbConext.getRowsAsync(tableSchemaName);
    return tableSchemaList;
  }

  async deleteTableSchemaAsync(tableName: string) {
    await this._idbConext.deleteRowAsync(tableSchemaName, tableName);
  }
} 
