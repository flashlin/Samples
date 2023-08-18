import initSqlJs, { type Database } from 'sql.js';
import wasmPath from '../assets/sql-wasm.wasm?url';
import { type IDataTable } from "./dataTypes";

//const wasmPath = import.meta.env.BASE_URL + 'assets/sql-wasm.wasm';
//const SQL = await initSqlJs({ locateFile: () => wasmPath });

type FetchRowFn = (row: any) => any;

export interface IColumnInfo {
    name: string;
    dataType: string;
}

let SQL: initSqlJs.SqlJsStatic | null = null;

export class SqliteDb {
    //SQL: initSqlJs.SqlJsStatic = null!;
    _db: Database = null!;

    async openAsync() {
        if (SQL == null) {
            SQL = await initSqlJs({ locateFile: () => wasmPath });
        }
        this._db = new SQL.Database();
    }

    execute(sql: string, data?: any) {
        console.log(sql, data);

        const db = this._db!;
        if (data != null) {
            const stmt = db.prepare(sql);
            stmt.bind(data);
            const success = stmt.step();
            console.log(success)
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

    query(sql: string, parameters?: initSqlJs.BindParams): IDataTable {
        const result: any[] = [];
        const columnNames = this.fetch(sql, (row) => {
            result.push(row);
        }, parameters);
        return {
            columnNames: columnNames,
            rows: result,
        };
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
        const columnNames = columns.map(x => `${x.name}`).join(', ');
        const values = columns.map(x => `:${x.name}`).join(', ');
        rows.forEach(row => {
            const insertQuery = `INSERT INTO ${tableName} (${columnNames}) VALUES(${values})`;
            const newData: any = {}
            for (const col of columns) {
                const key = col.name;
                newData[`:${key}`] = row[key];
            }
            this.execute(insertQuery, newData);

            // const values = columns.map(x => {
            //     if (x.dataType == 'TEXT') {
            //         return `'${row[x.name]}'`
            //     }
            //     return `${row[x.name]}`;
            // }).join(', ');
            // const insertQuery = `INSERT INTO ${tableName} (${columnNames}) VALUES(${values})`;
            // console.log(insertQuery)
            // this._db?.run(insertQuery)
            //const res = this._db?.exec("SELECT * FROM tb0");
            //console.log(res)
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
