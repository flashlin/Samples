import { IntellisenseItem } from "@/components/CodeEditorTypes"
import { IdbConext } from "./IdbKit"
import { AskLlmReq, useIntellisenseApi } from "./intellisenseApi";

const intellisenseApi = useIntellisenseApi();
async function askLlmAsync<T>(req: AskLlmReq): Promise<T> {
    console.info(req.instruction + ' ' + req.question);
    const resp = await intellisenseApi.askLlm(req);
    console.info(resp.answer);
    // 檢查 resp.answer 是否以 "```json" 開頭，如果是則移除開頭的 "```json"
    // 並且如果結尾是 "```" 則移除結尾的 "```"
    let answerText = resp.answer.trim();
    if (answerText.startsWith('```json')) {
        answerText = answerText.substring(7).trim();
    }
    if (answerText.endsWith('```')) {
        answerText = answerText.substring(0, answerText.length - 3).trim();
    }
    const answer = JSON.parse(answerText);
    return answer;
}

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

class LinqtsqlIntellisenseStorage {
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
    // 使用 GET 請求取得 /data/db_schema.json?t=${now}
    const response = await fetch(`/data/db_schema.json?t=${now}`, {
        method: 'GET'
    });
    if (!response.ok) {
        return [];
    }
    const data = await response.json();
    return data as DbSchema[];
}

export const dbSchemaJson = await getDbSchemaJsonAsync();

async function _empty(req: IntellisenseReq): Promise<IntellisenseResp> {
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
                getContext: () => 'FROM '
            }
        ]
    }
}

async function _from(req: IntellisenseReq): Promise<IntellisenseResp> {
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
            getContext: () => `${table} `
        }))
    }
}

async function _from_table_sel(req: IntellisenseReq): Promise<IntellisenseResp> {
    if( req.prevTokens.length === 0 ) {
        return {
            items: []
        }
    }
    const prevToken = req.prevTokens[req.prevTokens.length - 1];
    if( prevToken !== 'sel' ) {
        return {
            items: []
        }
    }
    return{
        items: [
            {
                title: 'select',
                getContext: () => 'ect '
            }
        ]
    }
}

interface TableAlias {
    table: string
    aliasName: string
}
async function _from_table_select(req: IntellisenseReq): Promise<IntellisenseResp> {
    if( req.prevTokens.length === 0 ) {
        return {
            items: []
        }
    }
    const prevToken = req.prevTokens[req.prevTokens.length - 1];
    if( prevToken !== 'select' ) {
        return {
            items: []
        }
    }
    // search tables from SQL (use LLM to fetch tables)
    const prompt = `content:
    ${req.prevText}

instruction:
以上是 SQL 語句，請你分析出 SQL 語句中使用的 tables 和 alias names，並回傳 tables 和 alias names 列表。
回傳 JSON 格式為：
[
    {
        table: string,
        aliasName: string
    }
]
    `
    const userQueryTableList = await askLlmAsync<TableAlias[]>({
        user: 'support',
        instruction: '',
        question: prompt,
        model_name: 'gemma-3n-e4b-it-text'
    });

    // search query SQL from HistoryDB by req.dbName and tables
    // if query SQL is found, use LLM fetch fields from query History then return

    // search fields of tables from tableSchemas
    const tableSchema = dbSchemaJson.find(db => db.dbName === req.dbName)
        ?.tables.find(table => table.tableName === userQueryTableList[0].table);

    // 依照 tableName 排序 tableSchema
    let sortedTableSchema = undefined;
    if (tableSchema) {
        // 如果 tableSchema 是陣列，排序；否則包成陣列再排序
        const schemas = Array.isArray(tableSchema) ? tableSchema : [tableSchema];
        sortedTableSchema = schemas.sort((a, b) => {
            if (a.tableName < b.tableName) return -1;
            if (a.tableName > b.tableName) return 1;
            return 0;
        });
    }
    // check tables 是否有 alias name ?
    // if alias name is found, use alias name + '.' + field name
    let fieldList: string[] = [];
    if( sortedTableSchema ) {
        for(let table of sortedTableSchema) {
            for(let field of table.fields) {
                const aliasName = userQueryTableList.find(t => t.table === table.tableName)?.aliasName ?? table.tableName;
                fieldList.push(`${aliasName}.${field.fieldName}`);
            }
        }
    }
    fieldList.sort();

    const title = fieldList.length > 9
        ? fieldList.slice(0, 9).join(', ') + '...'
        : fieldList.join(', ');
    const context = fieldList.join(', ');
    const getContext = () => {
        return context;
    }
    return {
        items: [
            {
                title: title,
                getContext: getContext
            }
        ]
    }
}


const _intellisenseArr: Array<(req: IntellisenseReq) => Promise<IntellisenseResp>> = [
    _empty,
    _from,
    _from_table_sel,
    _from_table_select
];

export async function provideIntellisenseAsync(req: IntellisenseReq): Promise<IntellisenseResp> {
    for(let item of _intellisenseArr) {
        const resp = await item(req);
        if( resp.items.length > 0 ) {
            return resp;
        }
    }
    return {
        items: [
            {
                title: '<No Result>',
                getContext: () => ''
            }
        ]
    }
}