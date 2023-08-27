import * as monaco from 'monaco-editor';
import { type ISuggestOption, type ICursorTextInfo, getTextByCursorPosition } from './monacoEx';
import { DatabaseOptions, type DatabaseOption } from './DatabaseOptions';
import { splitTableNameAndAlia } from './sqlLanguage';

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
