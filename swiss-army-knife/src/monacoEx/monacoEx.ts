import * as monaco from 'monaco-editor';


export interface IMonaco {
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
    languages: typeof monaco.languages;
}
// 重寫 monaco-editor 建議宣告
export interface ISuggestOption
    extends Pick<
        monaco.languages.CompletionItem, Exclude<keyof monaco.languages.CompletionItem, 'range'>
    > {
    range?: monaco.IRange |
    {
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
export const getTextByCursorPosition = (
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