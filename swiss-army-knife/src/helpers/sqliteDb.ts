import initSqlJs, { type Database } from 'sql.js';
import wasmPath from '../assets/sql-wasm.wasm?url';
//const wasmPath = import.meta.env.BASE_URL + 'assets/sql-wasm.wasm';
//const SQL = await initSqlJs({ locateFile: () => wasmPath });

type FetchRowFn = (row: initSqlJs.ParamsObject) => any;

export interface IColumnInfo {
    name: string;
    dataType: string;
}

export class SqliteDb {
    SQL: initSqlJs.SqlJsStatic = null!;
    _db: Database = null!;

    async open() {
        this.SQL = await initSqlJs({ locateFile: () => wasmPath });
        this._db = new this.SQL.Database();
    }

    execute(sql: string, data?: any) {
        this._db?.exec(sql, data);
    }

    fetch(sql: string, parameters: initSqlJs.BindParams, fetchRow: FetchRowFn) {
        const db = this._db;
        //var stmt = db.prepare("SELECT * FROM test WHERE col1 BETWEEN $start AND $end");
        //stmt.getAsObject({$start:1, $end:1}); // {col1:1, col2:111}
        const stmt = db.prepare(sql);
        stmt.bind(parameters);
        while (stmt.step()) {
            fetchRow(stmt.getAsObject());
        }
    }

    query(sql: string, parameters: initSqlJs.BindParams) {
        const result: any[] = [];
        this.fetch(sql, parameters, (row) => {
            result.push(row);
        });
        return result;
    }

    close() {
        this._db?.close();
    }

    createTable(tableName: string, row: any) {
        const columns = this.createTableColumns(row);
        const declareColumnNames = columns.map((c) => `${c.name} ${c.dataType}`).join(', ');
        const stat = `CREATE TABLE IF NOT EXISTS ${tableName} (${declareColumnNames})`;
        this.execute(stat);
        return columns;
    }

    importTable(tableName: string, rows: any[]) {
        const columns = this.createTable(tableName, rows[0]);
        const columnNames = columns.map(x => x.name).join(', ');
        const values = columns.map(x => {
            if (x.dataType == 'TEXT') {
                return "'${" + `${x.name}` + "}'"
            }
            return '${' + `${x.name}` + '}'
        }).join(', ');
        rows.forEach(row => {
            const insertQuery = `INSERT INTO ${tableName} (${columnNames}) VALUES(${values})`;
            this.execute(insertQuery, row);
        })
    }

    private createTableColumns(row: any) {
        const columns: IColumnInfo[] = [];
        for (const key in row) {
            if (typeof row[key] === 'number') {
                columns.push({
                    name: key,
                    dataType: 'NUMERIC',
                });
            } else {
                columns.push({
                    name: key,
                    dataType: 'TEXT',
                });
            }
        }
        return columns;
    }
}
