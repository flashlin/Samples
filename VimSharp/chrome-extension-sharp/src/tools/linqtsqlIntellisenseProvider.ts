import { IntellisenseItem } from "@/components/CodeEditorTypes"
import { IdbConext } from "./IdbKit"

interface IntellisenseReq {
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

function empty(req: IntellisenseReq): IntellisenseResp {
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

