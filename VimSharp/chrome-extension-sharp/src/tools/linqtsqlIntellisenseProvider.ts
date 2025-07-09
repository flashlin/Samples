import { IntellisenseItem } from "@/components/CodeEditorTypes"
import { IdbConext } from "./IdbKit"

interface IntellisenseReq {
    dbName: string
    prevTokens: string[]
    prevText: string
    nextTokens: string[]
    nextText: string
}

interface IntellisenseResp {
    items: IntellisenseItem[]
}


interface DatabaseTable {
    database: string
    tableName: string
}

class LinqtsqlIntellisenseStorege {
    private _idb: IdbConext;
    private _databaseTableList: DatabaseTable[] = [];
    constructor() {
        this._idb = new IdbConext();
    }
    async openAsync() {
        await this._idb.openAsync("linqtsqlIntellisense");
        const rows = await this._idb.getRowsAsync("databaseTable");
        this._databaseTableList = rows.map(row => ({
            database: row.database,
            tableName: row.tableName
        }));
    }
    async closeAsync() {
        await this._idb.closeAsync();
    }
    async getDatabaseTableListAsync(): Promise<DatabaseTable[]> {
        return this._databaseTableList;
    }
    async updateDatabaseTableListAsync(databaseTableList: DatabaseTable[]) {
        await this._idb.deleteTableAsync("databaseTable");
        for(let item in databaseTableList) {
            await this._idb.addRowAsync("databaseTable", item);
        }
    }
}


export interface FieldSchema {
    fieldName: string
    dataType: string
    size: number
    scaleSize: number
    isPrimaryKey: boolean
    isNull: boolean
    defaultValue: string | null
    isIdentity: boolean
    description: string
}

export interface TableSchema {
    tableName: string
    fields: FieldSchema[]
}

export interface DbSchema {
    dbName: string
    tables: TableSchema[]
}

async function getDbSchemaJsonAsync(): Promise<DbSchema[]> {
    const now = Date.now();
    const response = await fetch(`/data/db_schema.json?t=${now}`);
    if (!response.ok) {
        return [];
    }
    const data = await response.json();
    return data as DbSchema[];
}

const dbSchemaJson = await getDbSchemaJsonAsync();

function _empty(req: IntellisenseReq): IntellisenseResp {
    if( req.prevTokens.length !== 0 ) {
        return {
            items: []
        }
    }
    if( req.nextTokens.length !== 0 ) {
        return {
            items: []
        }
    }
    return {
        items: [
            {
                title: 'FROM',
                context: 'FROM '
            }
        ]
    }
}

function _from(req: IntellisenseReq): IntellisenseResp {
    if( req.prevTokens.length === 0 ) {
        return {
            items: []
        }
    }
    const prevToken = req.prevTokens[req.prevTokens.length - 1];
    if( prevToken.toUpperCase() !== 'FROM' ) {
        return {
            items: []
        }
    }
    const dbName = req.dbName;
    const dbSchema = dbSchemaJson.find(db => db.dbName === dbName);
    if( !dbSchema ) {
        return {
            items: []
        }
    }
    const tableList = dbSchema.tables.map(table => table.tableName);
    return {
        items: tableList.map(table => ({
            title: table,
            context: `${table} `
        }))
    }
}

function provideIntellisense(req: IntellisenseReq): IntellisenseResp {
    
}