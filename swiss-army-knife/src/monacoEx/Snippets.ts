import * as monaco from 'monaco-editor';
import { language as Language } from 'monaco-editor/esm/vs/basic-languages/sql/sql.js';

export interface Monaco {
    // CancellationTokenSource: typeof monaco.CancellationTokenSource
    // Emitter: typeof monaco.Emitter<any>
    // KeyCode: typeof monaco.KeyCode
    // KeyMod: typeof monaco.KeyMod
    // MarkerSeverity: typeof monaco.MarkerSeverity
    // MarkerTag: typeof monaco.MarkerTag
    // Position: typeof monaco.Position
    // Range: typeof monaco.Range
    // Selection: typeof monaco.Selection
    // SelectionDirection: typeof monaco.SelectionDirection
    // Token: typeof monaco.Token
    // Uri: typeof monaco.Uri
    // editor: typeof monaco.editor
    languages: typeof monaco.languages
}

export type FieldOption = {
    fieldName: string;
    fieldType: string;
    fieldComment: string;
    databaseName: string;
    tableName: string;
};

export type TableOption = {
    tableName: string;
    tableComment: string;
    fieldOptions: Array<FieldOption>;
};

export type DatabaseOption = {
    databaseName: string;
    tableOptions: Array<TableOption>;
};

export const SortText = {
    Database: '0',
    Table: '1',
    Column: '2',
    Keyword: '3',
};

export class DatabaseOptions {
    databaseOptions: Array<DatabaseOption>;

    constructor(databaseOptions: Array<DatabaseOption>) {
        this.databaseOptions = databaseOptions;
    }

    findDatabase(tokenNoDot: string) {
        const databaseOption = this.databaseOptions.find(
            (databaseOption: DatabaseOption) =>
                databaseOption.databaseName.toLowerCase() ===
                tokenNoDot.replace(/^.*,/g, ''),
        );
        return databaseOption;
    }

    /**
     * @param { string } databaseName
     * @returns { Array<SuggestOption> } []
     */
    getTableOptionsSuggestByDatabaseName = (databaseName: string): Array<SuggestOption> => {
        const currentDatabase = this.databaseOptions.find(
            (databaseOption: DatabaseOption) =>
                databaseOption.databaseName.toLowerCase() === databaseName,
        );
        // console.log("currentDatabase", currentDatabase, databaseName);
        return currentDatabase
            ? currentDatabase.tableOptions.map((tableOption: TableOption) => ({
                label: tableOption.tableName || '',
                kind: monaco.languages.CompletionItemKind.Struct,
                detail: `<Table> ${currentDatabase.databaseName} ${tableOption.tableComment || ''}`,
                sortText: SortText.Table,
                insertText: tableOption.tableName || '',
                documentation: tableOption.tableComment || '',
            }))
            : [];
    };
}

// 重寫 monaco-editor 建議宣告
export interface SuggestOption
    extends Pick<
        monaco.languages.CompletionItem,
        Exclude<keyof monaco.languages.CompletionItem, 'range'>
    > {
    range?:
    | monaco.IRange
    | {
        insert: monaco.IRange;
        replace: monaco.IRange;
    };
}



export interface ICursorTextInfo {
    textBeforePointer: string;
    textBeforePointerMulti: string;
    textAfterPointer: string;
    textAfterPointerMulti: string;
}

/**
 * @param { monaco.editor.ITextModel } model
 * @param { monaco.Position } position
 * @returns {
 *  textBeforePointer: 游標前面當行 Text
 *  textBeforePointerMulti: 游標前第一行到游標位置所有的 Text 
 *  textAfterPointer: 游標後當行 Text
 *  textAfterPointerMulti: 游標後到最後一行所有的 Text
 * }
 */
const getTextByCursorPosition = (
    model: monaco.editor.ITextModel,
    position: monaco.Position,
): ICursorTextInfo => {
    const { lineNumber, column } = position;

    const textBeforePointer = model.getValueInRange({
        startLineNumber: lineNumber,
        startColumn: 0,
        endLineNumber: lineNumber,
        endColumn: column,
    });

    const textBeforePointerMulti = model.getValueInRange({
        startLineNumber: 1,
        startColumn: 0,
        endLineNumber: lineNumber,
        endColumn: column,
    });

    const textAfterPointer = model.getValueInRange({
        startLineNumber: lineNumber,
        startColumn: column,
        endLineNumber: lineNumber,
        endColumn: model.getLineMaxColumn(model.getLineCount()),
    });

    const textAfterPointerMulti = model.getValueInRange({
        startLineNumber: lineNumber,
        startColumn: column,
        endLineNumber: model.getLineCount(),
        endColumn: model.getLineMaxColumn(model.getLineCount()),
    });

    return {
        textBeforePointer,
        textBeforePointerMulti,
        textAfterPointer,
        textAfterPointerMulti,
    };
};

interface ICursorContext extends ICursorTextInfo {
    textBeforeLastToken: string;
    state: any;
}

interface ISuggestionsHandler {
    setNext(handler: ISuggestionsHandler): ISuggestionsHandler;
    handle(context: ICursorContext): SuggestOption[];
}

abstract class AbstractSuggestionsHandler implements ISuggestionsHandler {
    private nextHandler: ISuggestionsHandler | undefined;
    public setNext(handler: ISuggestionsHandler): ISuggestionsHandler {
        this.nextHandler = handler;
        return handler;
    }
    public handle(context: ICursorContext): SuggestOption[] {
        if (this.nextHandler) {
            return this.nextHandler.handle(context);
        }
        return [];
    }
}

const createChain = (...handlers: ISuggestionsHandler[]): ISuggestionsHandler => {
    for (let i = 0; i < handlers.length - 1; i++) {
        handlers[i].setNext(handlers[i + 1]);
    }
    return handlers[0];
};

class Maybe_Database_Table_Field extends AbstractSuggestionsHandler {
    public handle(context: ICursorContext): SuggestOption[] {
        const {
            textBeforePointer,
        } = context;
        const textBeforeTokens = textBeforePointer.trim().split(/\s+/);
        const textBeforeLastToken = textBeforeTokens[textBeforeTokens.length - 1].toLowerCase();
        const newContext = {
            ...context,
            textBeforeLastToken: textBeforeLastToken
        };
        if (!textBeforeLastToken.endsWith('.')) {
            return super.handle(newContext);
        }

        // 如果最后一个文本后面 包含. 判断这个点之前的内容是否是database
        const textBeforeLastTokenNoDot = textBeforeLastToken.slice(0, textBeforeLastToken.length - 1);
        const databaseOptions: DatabaseOptions = context.state;

        const databaseOption = databaseOptions.findDatabase(textBeforeLastTokenNoDot);
        if (!databaseOption) {
            return super.handle(newContext);
        }

        const databaseName = textBeforeLastTokenNoDot.replace(/^.*,/g, '');
        return databaseOptions.getTableOptionsSuggestByDatabaseName(databaseName);
    }
}




export class SqlSnippets {
    private monaco: Monaco;
    // 自定义关键字
    private customKeywords: Array<string>;
    private databaseKeywords: Array<string>;
    private databaseOptions: Array<DatabaseOption>;
    private sortText: SortText;

    constructor(customKeywords?: Array<string>, databaseOptions?: Array<DatabaseOption>) {
        this.sortText = {
            Database: '0',
            Table: '1',
            Column: '2',
            Keyword: '3',
        };

        // 记录自定义关键字
        this.customKeywords = customKeywords || [];
        // 数据库关键字 将自定义关键字也柔和进去
        this.databaseKeywords = [
            ...Language.keywords,
            ...Language.operators,
            ...Language.builtinFunctions,
            ...this.customKeywords,
        ];
        // 记录数据库选项
        this.databaseOptions = databaseOptions || [];
        this.monaco = monaco;
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
        const {
            textBeforePointer,
            textBeforePointerMulti,
            // textAfterPointer,
            textAfterPointerMulti,
        } = getTextByCursorPosition(model, position);

        // const nextTokens = textAfterPointer.trim().split(/\s+/)
        // const nextToken = nextTokens[0].toLowerCase()
        // 獲取游標當前行所有的 sql 並且去掉前後空格
        const textBeforeTokens = textBeforePointer.trim().split(/\s+/);

        // 游標前最後一个 Word
        const textBeforeLastToken = textBeforeTokens[textBeforeTokens.length - 1].toLowerCase();

        // 有可能是 Database.Table.Field
        // 有可能是 Table.Field
        if (textBeforeLastToken.endsWith('.')) {
            // 如果最后一个文本后面 包含. 判断这个点之前的内容是否是database
            const textBeforeLastTokenNoDot = textBeforeLastToken.slice(0, textBeforeLastToken.length - 1);

            // 是否是数据库
            const databaseOption = this.databaseOptions.find(
                (databaseOption: DatabaseOption) =>
                    databaseOption.databaseName.toLowerCase() ===
                    textBeforeLastTokenNoDot.replace(/^.*,/g, ''),
            );

            if (databaseOption) {
                // 如果是数据库 就获取当前database 下面的 tableOptions
                const databaseName = textBeforeLastTokenNoDot.replace(/^.*,/g, '');
                return {
                    suggestions: [...this.getTableOptionsSuggestByDatabaseName(databaseName)],
                };
            } else if (
                this.getTableNameAndTableAlia(
                    textBeforePointerMulti.split(';')[textBeforePointerMulti.split(';').length - 1] +
                    textAfterPointerMulti.split(';')[0],
                )
            ) {
                const tableInfoList = this.getTableNameAndTableAlia(
                    textBeforePointerMulti.split(';')[textBeforePointerMulti.split(';').length - 1] +
                    textAfterPointerMulti.split(';')[0],
                );
                const currentTable = tableInfoList.find(
                    (item) => item.tableAlia === textBeforeLastTokenNoDot.replace(/^.*,/g, ''),
                );
                // <别名>.<字段> 联想
                if (currentTable && currentTable.tableName) {
                    return {
                        suggestions: this.getFieldOptionsSuggestByTableAlia(),
                    };
                } else {
                    return {
                        suggestions: [],
                    };
                }
            } else {
                return {
                    suggestions: [],
                };
            }
        } else if (
            textBeforeLastToken === 'from' ||
            textBeforeLastToken === 'join' ||
            /(from|join)\s+.*?\s?,\s*$/.test(textBeforePointer.replace(/.*?\(/gm, '').toLowerCase())
        ) {
            // 库名联想
            const dataBaseOptionsSuggest = this.getDatabaseOptionsSuggestions();
            return {
                suggestions: dataBaseOptionsSuggest,
            };
        } else if (
            [
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
                !this.databaseOptions.find(
                    (databaseOption: DatabaseOption) =>
                        `${databaseOption.databaseName}.` === textBeforeLastToken,
                )) ||
            /(select|where|order by|group by|by|and|or|having|distinct|on)\s+.*?\s?,\s*$/.test(
                textBeforePointer.toLowerCase(),
            )
        ) {
            // 字段联想
            return {
                suggestions: this.getFieldOptionsSuggestions(),
            };
        }

        // else if (this.customKeywords.toString().includes(textBeforeLastToken)) {
        //     // 自定义字段联想
        //     return {
        //         suggestions: this.getCustomSuggestions(textBeforeLastToken.startsWith('$')),
        //     }
        // }
        else {
            // 默认联想 数据库联想、关键字联想、表联想
            return {
                suggestions: [
                    ...this.getDatabaseOptionsSuggestions(),
                    ...this.getTableOptionsSuggestions(),
                    ...this.getKeywordOptionsSuggestions(),
                ],
            };
        }
    }

    /**
     * @desc 获取自定义联想建议
     */
    private getCustomSuggestions(startsWith$: boolean): Array<SuggestOption> {
        return this.customKeywords.map((customKeyword) => ({
            label: customKeyword,
            kind: this.monaco.languages.CompletionItemKind.Keyword,
            detail: '自定义联想',
            sortText: this.sortText.Keyword,
            insertText: startsWith$ ? customKeyword.slice(1) : customKeyword,
        }));
    }

    /**
     * @desc 获取数据库库名联想建议
     */
    getDatabaseOptionsSuggestions = (): Array<SuggestOption> => {
        return this.databaseOptions.map((databaseOption: DatabaseOption) => {
            return {
                label: databaseOption.databaseName || '',
                kind: this.monaco.languages.CompletionItemKind.Class,
                detail: `<数据库>`,
                sortText: this.sortText.Database,
                insertText: databaseOption.databaseName || '',
            };
        });
    };

    /**
     * 获取数据库关键字联想建议
     * @returns { Array<SuggestOption> } []
     */
    getKeywordOptionsSuggestions = (): Array<SuggestOption> =>
        this.databaseKeywords.map((databaseKeyword: string) => ({
            label: databaseKeyword,
            kind: this.monaco.languages.CompletionItemKind.Keyword,
            detail: '<关键字>',
            sortText: this.sortText.Keyword,
            insertText: databaseKeyword.startsWith('$') ? databaseKeyword.slice(1) : databaseKeyword,
        }));

    /**
     * 获取数据库表名建议
     * @return { Array<SuggestOption> } []
     */
    getTableOptionsSuggestions = (): Array<SuggestOption> => {
        const suggestOptions: Array<SuggestOption> = [];
        this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
            databaseOption.tableOptions.forEach((tableOption: TableOption) => {
                suggestOptions.push({
                    label: tableOption.tableName || '',
                    kind: this.monaco.languages.CompletionItemKind.Struct,
                    detail: `<Table> ${databaseOption.databaseName} ${tableOption.tableComment || ''}`,
                    sortText: this.sortText.Table,
                    insertText: tableOption.tableName || '',
                    documentation: tableOption.tableComment || '',
                });
            });
        });
        return suggestOptions;
    };

    /**
     * @param { string } databaseName
     * @returns { Array<SuggestOption> } []
     */
    getTableOptionsSuggestByDatabaseName = (databaseName: string): Array<SuggestOption> => {
        const currentDatabase = this.databaseOptions.find(
            (databaseOption: DatabaseOption) =>
                databaseOption.databaseName.toLowerCase() === databaseName,
        );
        // console.log("currentDatabase", currentDatabase, databaseName);
        return currentDatabase
            ? currentDatabase.tableOptions.map((tableOption: TableOption) => ({
                label: tableOption.tableName || '',
                kind: this.monaco.languages.CompletionItemKind.Struct,
                detail: `<Table> ${currentDatabase.databaseName} ${tableOption.tableComment || ''}`,
                sortText: this.sortText.Table,
                insertText: tableOption.tableName || '',
                documentation: tableOption.tableComment || '',
            }))
            : [];
    };

    /**
     * 获取所有表字段
     * @return {Array<SuggestOption>} []
     */
    getFieldOptionsSuggestions = (): Array<SuggestOption> => {
        const defaultFieldOptions: Array<SuggestOption> = [];
        this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
            databaseOption.tableOptions.forEach((tableOption: TableOption) => {
                tableOption.fieldOptions &&
                    tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
                        defaultFieldOptions.push({
                            label: fieldOption.fieldName || '',
                            kind: this.monaco.languages.CompletionItemKind.Field,
                            detail: `<Field> ${fieldOption.fieldComment || ''} <${fieldOption.fieldName}>`,
                            sortText: this.sortText.Column,
                            insertText: fieldOption.fieldName || '',
                            documentation: {
                                value: `
  ### Database: ${fieldOption.databaseName}
  ### Table: ${fieldOption.tableName}
  ### ${fieldOption.fieldComment || ''}`,
                            },
                        });
                    });
            });
        });
        return defaultFieldOptions;
    };

    /**
     * 根据别名获取所有表字段
     * @return {Array<SuggestOption>} []
     */
    getFieldOptionsSuggestByTableAlia = (): Array<SuggestOption> => {
        const defaultFieldOptions: Array<SuggestOption> = [];
        this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
            databaseOption.tableOptions.forEach((tableOption: TableOption) => {
                tableOption.fieldOptions &&
                    tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
                        defaultFieldOptions.push({
                            label: fieldOption.fieldName || '',
                            kind: this.monaco.languages.CompletionItemKind.Field,
                            detail: `<字段> ${fieldOption.fieldComment || ''} <${fieldOption.fieldType}>`,
                            sortText: this.sortText.Column,
                            insertText: fieldOption.fieldName || '',
                            documentation: {
                                value: `
  ### Database: ${fieldOption.databaseName}
  ### Table: ${fieldOption.tableName}
  ### ${fieldOption.fieldComment || ''}`,
                            },
                        });
                    });
            });
        });
        return defaultFieldOptions;
    };

    /**
     * 获取sql中所有的表名和别名
     * @param {*} sqlText SQL字符串
     * @return {Array<{tableName: string, tableAlia: string }>} []
     */
    getTableNameAndTableAlia = (
        sqlText: string,
    ): Array<{
        tableName: string;
        tableAlia: string;
    }> => {
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
