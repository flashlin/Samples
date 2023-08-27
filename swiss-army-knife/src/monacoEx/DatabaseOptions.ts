import * as monaco from 'monaco-editor';
import { language as Language } from 'monaco-editor/esm/vs/basic-languages/sql/sql.js';
import type { ISuggestOption } from './monacoEx';

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
    // Database Keywords
    databaseKeywords = [
        ...Language.keywords,
        ...Language.operators,
        ...Language.builtinFunctions,
        //...this.customKeywords,
    ];

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
     * @desc Database Name Suggestions
     */
    getDatabaseOptionsSuggestions = (): Array<ISuggestOption> => {
        return this.databaseOptions.map((databaseOption: DatabaseOption) => {
            return {
                label: databaseOption.databaseName || '',
                kind: monaco.languages.CompletionItemKind.Class,
                detail: `<Database>`,
                sortText: SortText.Database,
                insertText: databaseOption.databaseName || '',
            };
        });
    };

    /**
     * @param { string } databaseName
     * @returns { Array<ISuggestOption> } []
     */
    getTableOptionsSuggestByDatabaseName = (databaseName: string): Array<ISuggestOption> => {
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

    /**
         * 根据别名获取所有表字段
         * @return {Array<ISuggestOption>} []
         */
    getFieldOptionsSuggestByTableAlia = (): Array<ISuggestOption> => {
        const defaultFieldOptions: Array<ISuggestOption> = [];
        this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
            databaseOption.tableOptions.forEach((tableOption: TableOption) => {
                tableOption.fieldOptions &&
                    tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
                        defaultFieldOptions.push({
                            label: fieldOption.fieldName || '',
                            kind: monaco.languages.CompletionItemKind.Field,
                            detail: `<Field> ${fieldOption.fieldComment || ''} <${fieldOption.fieldType}>`,
                            sortText: SortText.Column,
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
     * 取得所有 Field
     * @return {Array<ISuggestOption>} []
     */
    getFieldOptionsSuggestions(): Array<ISuggestOption> {
        const defaultFieldOptions: Array<ISuggestOption> = [];
        this.databaseOptions.forEach((databaseOption: DatabaseOption) => {
            databaseOption.tableOptions.forEach((tableOption: TableOption) => {
                tableOption.fieldOptions &&
                    tableOption.fieldOptions.forEach((fieldOption: FieldOption) => {
                        defaultFieldOptions.push({
                            label: fieldOption.fieldName || '',
                            kind: monaco.languages.CompletionItemKind.Field,
                            detail: `<Field> ${fieldOption.fieldComment || ''} <${fieldOption.fieldName}>`,
                            sortText: SortText.Column,
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
    }

    /**
     * Database Keywords Sugg 
     * @returns { Array<ISuggestOption> } []
     */
    getKeywordOptionsSuggestions = (): Array<ISuggestOption> =>
        this.databaseKeywords.map((databaseKeyword: string) => ({
            label: databaseKeyword,
            kind: monaco.languages.CompletionItemKind.Keyword,
            detail: '<Keywords>',
            sortText: SortText.Keyword,
            insertText: databaseKeyword.startsWith('$') ? databaseKeyword.slice(1) : databaseKeyword,
        }));
}