import initSqlJs, { type Database } from 'sql.js';
import wasmPath from '../assets/sql-wasm.wasm?url';
import { type IDataFieldInfo, type IDataTable, type IMasterDetailDataTable, type IMergeTableForm } from './dataTypes';
import { getObjectKeys, zip } from './dataHelper';

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

    saveToLocalstorageAsync(key: string) {
        return new Promise(resolve => {
            const db = this._db!;
            const data = db.export();   // Uint8Array 格式
            const blob = new Blob([data], { type: 'application/octet-stream' });
            const reader = new FileReader();
            reader.onload = function (event: ProgressEvent<FileReader>) {
                const base64Blob = event.target!.result as string;
                localStorage.setItem(key, base64Blob);
                resolve(base64Blob);
            };
            reader.readAsDataURL(blob);
        });
    }

    async loadFromLocalStoreageAsync(key: string) {
        const base64Blob = localStorage.getItem(key);
        if (!base64Blob) {
            return;
        }

        // const binaryString = atob(base64Blob);
        // const uint8Array = new Uint8Array(binaryString.length);
        // for (let i = 0; i < binaryString.length; i++) {
        //     uint8Array[i] = binaryString.charCodeAt(i);
        // }
        const blob = this.base64ToBlob(base64Blob);
        const uint8Array = await this.blobToUint8ArrayAsync(blob);

        const newDb = new SQL!.Database(uint8Array);
        this._db = newDb;
    }

    private base64ToBlob(base64Blob: string) {
        const parts = base64Blob.split(',');
        const contentType = parts[0].split(':')[1];
        const byteCharacters = atob(parts[1]);
        const byteArrays = new Uint8Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteArrays[i] = byteCharacters.charCodeAt(i);
        }
        const restoredBlob = new Blob([byteArrays], { type: contentType });
        return restoredBlob;
    }

    private blobToUint8ArrayAsync(blob: Blob): Promise<Uint8Array> {
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = function (event: ProgressEvent<FileReader>) {
                const uint8Array = new Uint8Array(event!.target!.result as any);
                resolve(uint8Array);
            };
            reader.readAsArrayBuffer(blob);
        });
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
        const table = this.getAllTableNamesTable();
        return table.rows.map((row: any) => {
            return row.tableName;
        });
    }

    private getAllTableNamesTable(): IDataTable {
        const db = this._db;
        const sql = `SELECT name as tableName FROM sqlite_master WHERE type='table'`;
        return db.queryDataTable(sql);
    }

    getTableFieldsInfoTable(tableName: string): IDataTable {
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

    getTableFieldsInfo(tableName: string): IDataFieldInfo[] {
        const fieldInfoTable = this.getTableFieldsInfoTable(tableName);
        return fieldInfoTable.rows;
    }

    getAllTables(): IMasterDetailDataTable {
        const tableNamesTable = this.getAllTableNamesTable();
        const tableNames = tableNamesTable.rows.map((row: any) => {
            return row.tableName;
        });
        const tableInfosTable: IDataTable[] = [];
        for (const tableName of tableNames) {
            const tableInfo = this.getTableFieldsInfoTable(tableName);
            tableInfosTable.push(tableInfo);
        }
        return {
            master: tableNamesTable,
            detail: tableInfosTable
        };
    }

    mergeTable(req: IMergeTableForm) {
        console.log('merge', req);
        const tb1Columns = this.getTableFieldsInfo(req.table1.name);
        const tb2Columns = this.getTableFieldsInfo(req.table2.name);

        const tb1ColumnNames = tb1Columns.map(column => `${column.name}`);
        const tb2ColumnNames = tb2Columns.map(column => `${column.name}`);

        const columns = [
            ...this.addPrefix(req.table1.name, tb1ColumnNames, tb2ColumnNames),
            ...this.addPrefix(req.table2.name, tb2ColumnNames, tb1ColumnNames)
        ];
        const columnsStr = columns.join(',');
        const joinOnColumns = zip(req.table1.joinOnColumns, req.table2.joinOnColumns)
            .map(([column1, column2]) => {
                return `${req.table1.name}.${column1} = ${req.table2.name}.${column2}`;
            });
        const joinOnColumnsStr = joinOnColumns.join(" AND ");
        const insertColumns = [
            ...this.addPrefixInsert(req.table1.name, tb1ColumnNames, tb2ColumnNames),
            ...this.addPrefixInsert(req.table2.name, tb2ColumnNames, tb1ColumnNames),
        ].join(",");


        const sql1 = `SELECT ${columnsStr} FROM ${req.table1.name} JOIN ${req.table2.name} ON ${joinOnColumnsStr} limit 1`;
        console.log("first", sql1)
        const firstRow = this._db.query(sql1);
        console.log('get first', firstRow)
        this._db.importTable(req.name, firstRow);

        const sql2 = 'DELETE TABLE FROM ${req.name}';
        this._db.execute(sql2);
        console.log(sql2);

        const sql = `INSERT INTO ${req.name}(${insertColumns}) 
        SELECT ${columnsStr} FROM ${req.table1.name} JOIN ${req.table2.name} ON ${joinOnColumnsStr}`;
        this._db.execute(sql);
    }

    addPrefix(tableName: string, names: string[], otherNames: string[]) {
        const result = [];
        for (let n = 0; n < names.length; n++) {
            const name = names[n];
            if (otherNames.includes(name)) {
                result.push(`${tableName}.${name} as ${tableName}_${name}`);
            } else {
                result.push(`${tableName}.${name} as ${name}`);
            }
        }
        return result;
    }

    addPrefixInsert(tableName: string, names: string[], otherNames: string[]) {
        const result = [];
        for (let n = 0; n < names.length; n++) {
            const name = names[n];
            if (otherNames.includes(name)) {
                result.push(`${tableName}_${name}`);
            } else {
                result.push(`${name}`);
            }
        }
        return result;
    }
}
