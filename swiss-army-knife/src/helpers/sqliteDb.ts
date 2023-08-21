import initSqlJs, { type Database } from 'sql.js';
import wasmPath from '../assets/sql-wasm.wasm?url';
import { type IDataTable, type IDataTableNested } from './dataTypes';
import { getObjectKeys } from './dataHelper';

//const wasmPath = import.meta.env.BASE_URL + 'assets/sql-wasm.wasm';
//const SQL = await initSqlJs({ locateFile: () => wasmPath });

type FetchRowFn = (row: any) => any;

export interface IColumnInfo {
    name: string;
    dataType: string;
}

let SQL: initSqlJs.SqlJsStatic | null = null;

export class SqliteDb {
    _db: Database = null!;

    async openAsync() {
        if (SQL == null) {
            SQL = await initSqlJs({ locateFile: () => wasmPath });
        }
        this._db = new SQL.Database();
    }

    execute(sql: string, data?: initSqlJs.BindParams) {
        const db = this._db!;
        if (data != null) {
            const stmt = db.prepare(sql);
            stmt.bind(data);
            stmt.step();
            stmt.free();
            return;
        }

        db.run(sql);
    }

    /**
     *
     * @param sql "select name from customer where name=:name"
     * @param fetchRow
     * @param parameters {":name":"flash"}
     * @returns columnNames ["name"]
     */
    fetch(sql: string, fetchRow: FetchRowFn, parameters?: initSqlJs.BindParams) {
        const db = this._db;
        const stmt = db.prepare(sql);
        if (parameters != null) {
            stmt.bind(parameters);
        }
        while (stmt.step()) {
            fetchRow(stmt.getAsObject());
        }
        const columnNames = stmt.getColumnNames();
        return columnNames;
    }

    queryDataTableByBindParams(
        sql: string,
        parameters?: initSqlJs.BindParams
    ): IDataTable {
        const result: any[] = [];
        const columnNames = this.fetch(
            sql,
            (row) => {
                result.push(row);
            },
            parameters
        );
        return {
            columnNames: columnNames,
            rows: result
        };
    }

    queryDataTable(sql: string, data?: any): IDataTable {
        const result: any[] = [];
        const columnNames = this.fetch(
            sql,
            (row) => {
                result.push(row);
            },
            this.toQueryParameters(data)
        );
        return {
            columnNames: columnNames,
            rows: result
        };
    }

    query<T extends object>(sql: string, data?: any): T[] {
        const result: T[] = [];
        this.fetch(
            sql,
            (row) => {
                result.push(row);
            },
            this.toQueryParameters(data)
        );
        return result;
    }

    close() {
        this._db?.close();
    }

    saveToLocalstorage(key: string) {
        const db = this._db!;
        const data = db.export();   // Uint8Array 格式
        const blob = new Blob([data], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        localStorage.setItem(key , url);
    }

    async loadFromLocalStoreage(key: string) {
        const savedUrl = localStorage.getItem(key);
        if (!savedUrl) {
            return;
        }
        const response = await fetch(savedUrl);
        const data = await response.arrayBuffer();
        const newDb = new SQL!.Database(new Uint8Array(data));
        this._db = newDb;
    }

    public dropTable(tableName: string) {
        const sql = `DROP TABLE IF EXISTS ${tableName}`;
        this.execute(sql);
    }

    private createTable(tableName: string, row: any) {
        const columns = this.createTableColumns(row);
        const declareColumnNames = columns
            .map((c) => `${c.name} ${c.dataType}`)
            .join(', ');
        const stat = `CREATE TABLE IF NOT EXISTS ${tableName} (${declareColumnNames})`;
        this.execute(stat);
        return columns;
    }

    importTable(tableName: string, rows: any[]): number {
        const columns = this.createTable(tableName, rows[0]);
        const columnNames = columns.map((x) => `${x.name}`).join(', ');
        const values = columns.map((x) => `:${x.name}`).join(', ');
        let count = 0;
        rows.forEach((row) => {
            const insertQuery = `INSERT INTO ${tableName} (${columnNames}) VALUES(${values})`;
            const newData: any = {};
            for (const col of columns) {
                const key = col.name;
                newData[`:${key}`] = row[key];
            }
            this.execute(insertQuery, newData);
            count++;
        });
        return count;
    }

    private toQueryParameters(obj?: any) {
        if (obj == null) {
            return null;
        }
        const queryParameters: initSqlJs.BindParams = {};
        for (const key of getObjectKeys(obj!)) {
            queryParameters[`:${key}`] = obj[key];
        }
        return queryParameters;
    }

    private createTableColumns(row: any) {
        const columns: IColumnInfo[] = [];
        for (const key in row) {
            if (typeof row[key] === 'number') {
                columns.push({
                    name: key,
                    dataType: 'NUMERIC'
                });
            } else {
                columns.push({
                    name: key,
                    dataType: 'TEXT'
                });
            }
        }
        return columns;
    }
}

export class QuerySqliteService {
    _db: SqliteDb;
    constructor(db: SqliteDb) {
        this._db = db;
    }

    getAllTableNames(): string[] {
        const table = this.getAllTableNamesToDataTable();
        return table.rows.map((row: any) => {
            return row.tableName;
        });
    }

    private getAllTableNamesToDataTable(): IDataTable {
        const db = this._db;
        const sql = `SELECT name as tableName FROM sqlite_master WHERE type='table'`;
        return db.queryDataTable(sql);
    }

    getTableFieldsInfo(tableName: string): IDataTable {
        const db = this._db;
        const sql = `SELECT [pk], [name], [type], [notnull] FROM pragma_table_info(:tableName)`;
        const result = db.queryDataTableByBindParams(sql, {
            ':tableName': tableName
        });
        result.columnNames = ['name', 'dataType', 'isPrimaryKey', 'isNullable'];
        result.rows = result.rows.map((row: any) => {
            return {
                name: row.name,
                dataType: row.type,
                isPrimaryKey: row.pk == 1,
                isNullable: row.notnull == 0
            };
        });
        return result;
    }

    getAllTables(): IDataTableNested {
        const tableNamesTable = this.getAllTableNamesToDataTable();
        const tableNames = tableNamesTable.rows.map((row: any) => {
            return row.tableName;
        });
        const tableInfosTable: IDataTable[] = [];
        for (const tableName of tableNames) {
            const tableInfo = this.getTableFieldsInfo(tableName);
            tableInfosTable.push(tableInfo);
        }
        return {
            master: tableNamesTable,
            detail: tableInfosTable
        };
    }
}
