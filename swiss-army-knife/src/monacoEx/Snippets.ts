import * as monaco from 'monaco-editor';
import { type ISuggestOption, type ICursorTextInfo, getTextByCursorPosition } from './monacoEx';
import { DatabaseOptions, type DatabaseOption } from './DatabaseOptions';

interface ICursorContext<T> extends ICursorTextInfo {
    textBeforeLastToken: string;
    state: T;
}

interface ISuggestionsHandler<T> {
    setNext(handler: ISuggestionsHandler<T>): ISuggestionsHandler<T>;
    handle(context: ICursorContext<T>): ISuggestOption[];
}

abstract class AbstractSuggestionsHandler<T> implements ISuggestionsHandler<T> {
    private nextHandler: ISuggestionsHandler<T> | undefined;
    public setNext(handler: ISuggestionsHandler<T>): ISuggestionsHandler<T> {
        this.nextHandler = handler;
        return handler;
    }
    public handle(context: ICursorContext<T>): ISuggestOption[] {
        if (this.nextHandler) {
            return this.nextHandler.handle(context);
        }
        return [];
    }
}

const createChain = <T>(...handlers: ISuggestionsHandler<T>[]): ISuggestionsHandler<T> => {
    for (let i = 0; i < handlers.length - 1; i++) {
        handlers[i].setNext(handlers[i + 1]);
    }
    return handlers[0];
};

/////////////////////////////////////////////////
export interface ITableNameAlia {
    tableName: string;
    tableAlia: string;
}
/**
 * 獲取 sql 中所有的 TableName 和 Alia
 * @param {*} sqlText SQL字符串
 * @return {Array<ITableNameAlia>} []
 */
export const splitTableNameAndAlia = (sqlText: string): Array<ITableNameAlia> => {
    const regTableAliaFrom =
        /(^|(\s+))from\s+([^\s]+(\s+|(\s+as\s+))[^\s]+(\s+|,)\s*)+(\s+(where|left|right|full|join|inner|union))?/gi;

    const regTableAliaJoin = /(^|(\s+))join\s+([^\s]+)\s+(as\s+)?([^\s]+)\s+on/gi;

    const regTableAliaFromList = sqlText.match(regTableAliaFrom) || [];

    const regTableAliaJoinList = sqlText.match(regTableAliaJoin) || [];

    const strList = [
        ...regTableAliaFromList.map((item) =>
            item
                .replace(/(^|(\s+))from\s+/gi, '')
                .replace(/\s+(where|left|right|full|join|inner|union)((\s+.*?$)|$)/gi, '')
                .replace(/\s+as\s+/gi, ' ')
                .trim(),
        ),
        ...regTableAliaJoinList.map((item) =>
            item
                .replace(/(^|(\s+))join\s+/gi, '')
                .replace(/\s+on((\s+.*?$)|$)/, '')
                .replace(/\s+as\s+/gi, ' ')
                .trim(),
        ),
    ];

    const tableList: Array<{
        tableName: string;
        tableAlia: string;
    }> = [];

    strList.map((tableAndAlia) => {
        tableAndAlia.split(',').forEach((item) => {
            const tableName = item.trim().split(/\s+/)[0];
            const tableAlia = item.trim().split(/\s+/)[1];
            tableList.push({
                tableName,
                tableAlia,
            });
        });
    });
    return tableList;
};

///////////////////////////////////////////////////////////////////////////////
class Maybe_Database_Table_Field extends AbstractSuggestionsHandler<DatabaseOptions> {
    public handle(context: ICursorContext<DatabaseOptions>): ISuggestOption[] {
        const {
            textBeforeLastToken
        } = context;

        if (!textBeforeLastToken.endsWith('.')) {
            return super.handle(context);
        }

        // 如果最后一个文本后面 包含. 判断这个点之前的内容是否是database
        const textBeforeLastTokenNoDot = textBeforeLastToken.slice(0, textBeforeLastToken.length - 1);
        const databaseOptions = context.state;

        const databaseOption = databaseOptions.findDatabase(textBeforeLastTokenNoDot);
        if (!databaseOption) {
            return super.handle(context);
        }

        const databaseName = textBeforeLastTokenNoDot.replace(/^.*,/g, '');
        return databaseOptions.getTableOptionsSuggestByDatabaseName(databaseName);
    }
}

class Maybe_Table_Field extends AbstractSuggestionsHandler<DatabaseOptions> {
    public handle(context: ICursorContext<DatabaseOptions>): ISuggestOption[] {
        const { textBeforePointerMulti, textAfterPointerMulti } = context;
        const tableInfoList = splitTableNameAndAlia(
            textBeforePointerMulti.split(';')[textBeforePointerMulti.split(';').length - 1] +
            textAfterPointerMulti.split(';')[0],
        )
        if (!tableInfoList) {
            return super.handle(context);
        }

        const { textBeforeLastToken, state } = context;
        const textBeforeLastTokenNoDot = textBeforeLastToken.slice(0, textBeforeLastToken.length - 1);
        const currentTable = tableInfoList.find(
            (item) => item.tableAlia === textBeforeLastTokenNoDot.replace(/^.*,/g, ''),
        );
        // <别名>.<字段> 联想
        if (currentTable && currentTable.tableName) {
            return state.getFieldOptionsSuggestByTableAlia();
        }
        return [];
    }
}

class Maybe_From extends AbstractSuggestionsHandler<DatabaseOptions> {
    public handle(context: ICursorContext<DatabaseOptions>): ISuggestOption[] {
        const { textBeforeLastToken, textBeforePointer } = context;

        if (
            textBeforeLastToken === 'from' ||
            textBeforeLastToken === 'join' ||
            /(from|join)\s+.*?\s?,\s*$/.test(textBeforePointer.replace(/.*?\(/gm, '').toLowerCase())
        ) {
            const { state } = context;
            return state.getDatabaseOptionsSuggestions();
        }

        return super.handle(context);
    }
}

class Maybe_Select extends AbstractSuggestionsHandler<DatabaseOptions> {
    public handle(context: ICursorContext<DatabaseOptions>): ISuggestOption[] {
        const { textBeforeLastToken, textBeforePointer, state } = context;

        if ([
            'select',
            'where',
            'order by',
            'group by',
            'by',
            'and',
            'or',
            'having',
            'distinct',
            'on',
        ].includes(textBeforeLastToken.replace(/.*?\(/g, '')) ||
            (textBeforeLastToken.endsWith('.') &&
                !state.databaseOptions.find(
                    (databaseOption: DatabaseOption) =>
                        `${databaseOption.databaseName}.` === textBeforeLastToken,
                )) ||
            /(select|where|order by|group by|by|and|or|having|distinct|on)\s+.*?\s?,\s*$/.test(
                textBeforePointer.toLowerCase(),
            )) {
            return state.getFieldOptionsSuggestions();
        }
        return super.handle(context);
    }
}

class Maybe_Default extends AbstractSuggestionsHandler<DatabaseOptions> {
    public handle(context: ICursorContext<DatabaseOptions>): ISuggestOption[] {
        const { state } = context;
        return [
            ...state.getDatabaseOptionsSuggestions(),
            ...state.getFieldOptionsSuggestions(),
            ...state.getKeywordOptionsSuggestions(),
        ];
    }
}



export class SqlSnippets {
    private customKeywords: Array<string>;
    private databaseOptions: Array<DatabaseOption>;

    constructor(customKeywords?: Array<string>, databaseOptions?: Array<DatabaseOption>) {
        // 记录自定义关键字
        this.customKeywords = customKeywords || [];
        // 数据库关键字 将自定义关键字也柔和进去
        this.databaseOptions = databaseOptions || [];
    }

    /**
     * 动态设置数据库表&&数据库字段
     * @param {*} databaseOptions 数据库数据
     * @example [{ databaseName: '', tableOptions: [{ tableName: '', fielsOptions: [ {  fieldName: "" ,fieldType: "" ,fieldComment: "" ,databaseName: "" ,tableName: ""  }] }] }]
     */
    private setDatabaseOption(databaseOptions: Array<DatabaseOption>): void {
        this.databaseOptions = databaseOptions;
    }

    /**
     * monaco提示方法
     * @param { monaco.editor.ITextModel } model
     * @param { monaco.Position } position
     */
    async provideCompletionItems(model: monaco.editor.ITextModel, position: monaco.Position) {
        const cursorInfo = getTextByCursorPosition(model, position);
        const { textBeforePointer } = cursorInfo;
        const textBeforeTokens = textBeforePointer.trim().split(/\s+/);
        const textBeforeLastToken = textBeforeTokens[textBeforeTokens.length - 1].toLowerCase();
        const newContext = {
            ...cursorInfo,
            textBeforeLastToken: textBeforeLastToken,
            state: new DatabaseOptions(this.databaseOptions)
        };

        const chain = createChain(
            new Maybe_Database_Table_Field(),
            new Maybe_Table_Field(),
            new Maybe_From(),
            new Maybe_Select(),
            new Maybe_Default()
        );
        const result = chain.handle(newContext);
        return {
            suggestions: result,
        };
    }
}






// export type SortText = {
//     Database: '0',
//     Table: '1',
//     Column: '2',
//     Keyword: '3',
// }

// export class SqlSnippets {
//     private monaco: Monaco;
//     // 自定义关键字
//     private customKeywords: Array<string>;
//     private databaseKeywords: Array<string>;
//     private databaseOptions: Array<DatabaseOption>;
//     private sortText: SortText;

//     constructor(customKeywords?: Array<string>, databaseOptions?: Array<DatabaseOption>) {
//         this.sortText = {
//             Database: '0',
//             Table: '1',
//             Column: '2',
//             Keyword: '3',
//         };

//         // 记录自定义关键字
//         this.customKeywords = customKeywords || [];
//         // 数据库关键字 将自定义关键字也柔和进去
//         this.databaseKeywords = [
//             ...Language.keywords,
//             ...Language.operators,
//             ...Language.builtinFunctions,
//             ...this.customKeywords,
//         ];
//         // 记录数据库选项
//         this.databaseOptions = databaseOptions || [];
//         this.monaco = monaco;
//     }

//     /**
//      * 动态设置数据库表&&数据库字段
//      * @param {*} databaseOptions 数据库数据
//      * @example [{ databaseName: '', tableOptions: [{ tableName: '', fielsOptions: [ {  fieldName: "" ,fieldType: "" ,fieldComment: "" ,databaseName: "" ,tableName: ""  }] }] }]
//      */
//     private setDatabaseOption(databaseOptions: Array<DatabaseOption>): void {
//         this.databaseOptions = databaseOptions;
//     }

//     /**
//      * monaco提示方法
//      * @param { monaco.editor.ITextModel } model
//      * @param { monaco.Position } position
//      */
//     async provideCompletionItems(model: monaco.editor.ITextModel, position: monaco.Position) {
//         const {
//             textBeforePointer,
//             textBeforePointerMulti,
//             // textAfterPointer,
//             textAfterPointerMulti,
//         } = getTextByCursorPosition(model, position);

//         // const nextTokens = textAfterPointer.trim().split(/\s+/)
//         // const nextToken = nextTokens[0].toLowerCase()
//         // 獲取游標當前行所有的 sql 並且去掉前後空格
//         const textBeforeTokens = textBeforePointer.trim().split(/\s+/);

//         // 游標前最後一个 Word
//         const textBeforeLastToken = textBeforeTokens[textBeforeTokens.length - 1].toLowerCase();

//         // 有可能是 Database.Table.Field
//         // 有可能是 Table.Field
//         if (textBeforeLastToken.endsWith('.')) {
//             // 如果最后一个文本后面 包含. 判断这个点之前的内容是否是database
//             const textBeforeLastTokenNoDot = textBeforeLastToken.slice(0, textBeforeLastToken.length - 1);

//             // 是否是数据库
//             const databaseOption = this.databaseOptions.find(
//                 (databaseOption: DatabaseOption) =>
//                     databaseOption.databaseName.toLowerCase() ===
//                     textBeforeLastTokenNoDot.replace(/^.*,/g, ''),
//             );

//             if (databaseOption) {
//                 // 如果是数据库 就获取当前database 下面的 tableOptions
//                 const databaseName = textBeforeLastTokenNoDot.replace(/^.*,/g, '');
//                 return {
//                     suggestions: [...this.getTableOptionsSuggestByDatabaseName(databaseName)],
//                 };
//             } else if (
//                 this.getTableNameAndTableAlia(
//                     textBeforePointerMulti.split(';')[textBeforePointerMulti.split(';').length - 1] +
//                     textAfterPointerMulti.split(';')[0],
//                 )
//             ) {
//                 const tableInfoList = this.getTableNameAndTableAlia(
//                     textBeforePointerMulti.split(';')[textBeforePointerMulti.split(';').length - 1] +
//                     textAfterPointerMulti.split(';')[0],
//                 );
//                 const currentTable = tableInfoList.find(
//                     (item) => item.tableAlia === textBeforeLastTokenNoDot.replace(/^.*,/g, ''),
//                 );
//                 // <别名>.<字段> 联想
//                 if (currentTable && currentTable.tableName) {
//                     return {
//                         suggestions: this.getFieldOptionsSuggestByTableAlia(),
//                     };
//                 } else {
//                     return {
//                         suggestions: [],
//                     };
//                 }
//             } else {
//                 return {
//                     suggestions: [],
//                 };
//             }
//         } else if (
//             textBeforeLastToken === 'from' ||
//             textBeforeLastToken === 'join' ||
//             /(from|join)\s+.*?\s?,\s*$/.test(textBeforePointer.replace(/.*?\(/gm, '').toLowerCase())
//         ) {
//             // 库名联想
//             const dataBaseOptionsSuggest = this.getDatabaseOptionsSuggestions();
//             return {
//                 suggestions: dataBaseOptionsSuggest,
//             };
//         } else if (
//             [
//                 'select',
//                 'where',
//                 'order by',
//                 'group by',
//                 'by',
//                 'and',
//                 'or',
//                 'having',
//                 'distinct',
//                 'on',
//             ].includes(textBeforeLastToken.replace(/.*?\(/g, '')) ||
//             (textBeforeLastToken.endsWith('.') &&
//                 !this.databaseOptions.find(
//                     (databaseOption: DatabaseOption) =>
//                         `${databaseOption.databaseName}.` === textBeforeLastToken,
//                 )) ||
//             /(select|where|order by|group by|by|and|or|having|distinct|on)\s+.*?\s?,\s*$/.test(
//                 textBeforePointer.toLowerCase(),
//             )
//         ) {
//             // 字段联想
//             return {
//                 suggestions: this.getFieldOptionsSuggestions(),
//             };
//         }

//         // else if (this.customKeywords.toString().includes(textBeforeLastToken)) {
//         //     // 自定义字段联想
//         //     return {
//         //         suggestions: this.getCustomSuggestions(textBeforeLastToken.startsWith('$')),
//         //     }
//         // }
//         else {
//             // 默认联想 数据库联想、关键字联想、表联想
//             return {
//                 suggestions: [
//                     ...this.getDatabaseOptionsSuggestions(),
//                     ...this.getTableOptionsSuggestions(),
//                     ...this.getKeywordOptionsSuggestions(),
//                 ],
//             };
//         }
//     }

//     /**
//      * @desc 获取自定义联想建议
//      */
//     private getCustomSuggestions(startsWith$: boolean): Array<SuggestOption> {
//         return this.customKeywords.map((customKeyword) => ({
//             label: customKeyword,
//             kind: this.monaco.languages.CompletionItemKind.Keyword,
//             detail: '自定义联想',
//             sortText: this.sortText.Keyword,
//             insertText: startsWith$ ? customKeyword.slice(1) : customKeyword,
//         }));
//     }

//     /**
//      * @desc 获取数据库库名联想建议
//      */
//     getDatabaseOptionsSuggestions = (): Array<SuggestOption> => {
//         return this.databaseOptions.map((databaseOption: DatabaseOption) => {
//             return {
//                 label: databaseOption.databaseName || '',
//                 kind: this.monaco.languages.CompletionItemKind.Class,
//                 detail: `<数据库>`,
//                 sortText: this.sortText.Database,
//                 insertText: databaseOption.databaseName || '',
//             };
//         });
//     };

//     /**
//      * 获取数据库关键字联想建议
//      * @returns { Array<SuggestOption> } []
//      */
//     getKeywordOptionsSuggestions = (): Array<SuggestOption> =>
//         this.databaseKeywords.map((databaseKeyword: string) => ({
//             label: databaseKeyword,
//             kind: this.monaco.languages.CompletionItemKind.Keyword,
//             detail: '<关键字>',
//             sortText: this.sortText.Keyword,
//             insertText: databaseKeyword.startsWith('$') ? databaseKeyword.slice(1) : databaseKeyword,
//         }));

//     /**
//      * 获取数据库表名建议
//      * @return { Array<SuggestOption> } []
//      */
//     getTableOptionsSuggestions = (): Array<SuggestOption> => {
//         const suggestOptions: Array<SuggestOption> = [];
//         this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
//             databaseOption.tableOptions.forEach((tableOption: TableOption) => {
//                 suggestOptions.push({
//                     label: tableOption.tableName || '',
//                     kind: this.monaco.languages.CompletionItemKind.Struct,
//                     detail: `<Table> ${databaseOption.databaseName} ${tableOption.tableComment || ''}`,
//                     sortText: this.sortText.Table,
//                     insertText: tableOption.tableName || '',
//                     documentation: tableOption.tableComment || '',
//                 });
//             });
//         });
//         return suggestOptions;
//     };

//     /**
//      * @param { string } databaseName
//      * @returns { Array<SuggestOption> } []
//      */
//     getTableOptionsSuggestByDatabaseName = (databaseName: string): Array<SuggestOption> => {
//         const currentDatabase = this.databaseOptions.find(
//             (databaseOption: DatabaseOption) =>
//                 databaseOption.databaseName.toLowerCase() === databaseName,
//         );
//         // console.log("currentDatabase", currentDatabase, databaseName);
//         return currentDatabase
//             ? currentDatabase.tableOptions.map((tableOption: TableOption) => ({
//                 label: tableOption.tableName || '',
//                 kind: this.monaco.languages.CompletionItemKind.Struct,
//                 detail: `<Table> ${currentDatabase.databaseName} ${tableOption.tableComment || ''}`,
//                 sortText: this.sortText.Table,
//                 insertText: tableOption.tableName || '',
//                 documentation: tableOption.tableComment || '',
//             }))
//             : [];
//     };

//     /**
//      * 获取所有表字段
//      * @return {Array<SuggestOption>} []
//      */
//     getFieldOptionsSuggestions = (): Array<SuggestOption> => {
//         const defaultFieldOptions: Array<SuggestOption> = [];
//         this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
//             databaseOption.tableOptions.forEach((tableOption: TableOption) => {
//                 tableOption.fieldOptions &&
//                     tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
//                         defaultFieldOptions.push({
//                             label: fieldOption.fieldName || '',
//                             kind: this.monaco.languages.CompletionItemKind.Field,
//                             detail: `<Field> ${fieldOption.fieldComment || ''} <${fieldOption.fieldName}>`,
//                             sortText: this.sortText.Column,
//                             insertText: fieldOption.fieldName || '',
//                             documentation: {
//                                 value: `
//   ### Database: ${fieldOption.databaseName}
//   ### Table: ${fieldOption.tableName}
//   ### ${fieldOption.fieldComment || ''}`,
//                             },
//                         });
//                     });
//             });
//         });
//         return defaultFieldOptions;
//     };

//     /**
//      * 根据别名获取所有表字段
//      * @return {Array<SuggestOption>} []
//      */
//     getFieldOptionsSuggestByTableAlia = (): Array<SuggestOption> => {
//         const defaultFieldOptions: Array<SuggestOption> = [];
//         this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
//             databaseOption.tableOptions.forEach((tableOption: TableOption) => {
//                 tableOption.fieldOptions &&
//                     tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
//                         defaultFieldOptions.push({
//                             label: fieldOption.fieldName || '',
//                             kind: this.monaco.languages.CompletionItemKind.Field,
//                             detail: `<字段> ${fieldOption.fieldComment || ''} <${fieldOption.fieldType}>`,
//                             sortText: this.sortText.Column,
//                             insertText: fieldOption.fieldName || '',
//                             documentation: {
//                                 value: `
//   ### Database: ${fieldOption.databaseName}
//   ### Table: ${fieldOption.tableName}
//   ### ${fieldOption.fieldComment || ''}`,
//                             },
//                         });
//                     });
//             });
//         });
//         return defaultFieldOptions;
//     };

//     /**
//      * 获取sql中所有的表名和别名
//      * @param {*} sqlText SQL字符串
//      * @return {Array<{tableName: string, tableAlia: string }>} []
//      */
//     getTableNameAndTableAlia = (
//         sqlText: string,
//     ): Array<{
//         tableName: string;
//         tableAlia: string;
//     }> => {
//         const regTableAliaFrom =
//             /(^|(\s+))from\s+([^\s]+(\s+|(\s+as\s+))[^\s]+(\s+|,)\s*)+(\s+(where|left|right|full|join|inner|union))?/gi;

//         const regTableAliaJoin = /(^|(\s+))join\s+([^\s]+)\s+(as\s+)?([^\s]+)\s+on/gi;

//         const regTableAliaFromList = sqlText.match(regTableAliaFrom) || [];

//         const regTableAliaJoinList = sqlText.match(regTableAliaJoin) || [];

//         const strList = [
//             ...regTableAliaFromList.map((item) =>
//                 item
//                     .replace(/(^|(\s+))from\s+/gi, '')
//                     .replace(/\s+(where|left|right|full|join|inner|union)((\s+.*?$)|$)/gi, '')
//                     .replace(/\s+as\s+/gi, ' ')
//                     .trim(),
//             ),
//             ...regTableAliaJoinList.map((item) =>
//                 item
//                     .replace(/(^|(\s+))join\s+/gi, '')
//                     .replace(/\s+on((\s+.*?$)|$)/, '')
//                     .replace(/\s+as\s+/gi, ' ')
//                     .trim(),
//             ),
//         ];

//         const tableList: Array<{
//             tableName: string;
//             tableAlia: string;
//         }> = [];

//         strList.map((tableAndAlia) => {
//             tableAndAlia.split(',').forEach((item) => {
//                 const tableName = item.trim().split(/\s+/)[0];
//                 const tableAlia = item.trim().split(/\s+/)[1];
//                 tableList.push({
//                     tableName,
//                     tableAlia,
//                 });
//             });
//         });
//         return tableList;
//     };
// }
