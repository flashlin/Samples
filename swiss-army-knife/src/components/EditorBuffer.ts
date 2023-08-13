import { Observable, fromEvent } from 'rxjs';
import { buffer, filter, map, scan } from 'rxjs/operators';

export class LineBuffer {
    lineNum = 0;
    cols = 0;
    tabWidth = 3;
    br = false;
    content = '';
    color = '';
    editorBuffer: EditorBuffer;

    constructor(editorBuffer: EditorBuffer, lineNum: number, cols: number) {
        this.editorBuffer = editorBuffer;
        this.lineNum = lineNum;
        this.cols = cols;
    }

    append(content: string) {
        if (content == undefined) {
            return;
        }

        let line = '';
        let n = 0;
        for (; n < content.length; n++) {
            const c = content[n];
            if (c === undefined) break;
            if (c == '\n') {
                n++;
                this.br = true;
                break;
            }
            line += c;
        }
        this.content += line;
        const remainingContent = content.substring(n);
        if (remainingContent.length == 0) {
            return;
        }
        this.editorBuffer.appendLine(this.lineNum + 1, 0, remainingContent);
    }

    insert(cols: number, content: string) {
        const lineContent = this.getContent();
        const maxCols = lineContent.length + this.cols;
        if (cols == maxCols) {
            this.append(content);
            return;
        }
        if (cols > maxCols) {
            throw new Error(`cols is too Large ${cols}`);
        }
        const prev = lineContent.substring(0, cols - this.cols);
        const after = lineContent.substring(cols);
        const newContent = content + after;
        this.content = prev;
        this.append(newContent);
    }

    delete(cols: number, length: number): number {
        const lineContent = this.getContent();
        if (cols >= lineContent.length + this.cols) {
            throw new Error(`${cols} cols is out of range`);
        }
        const prev = lineContent.substring(0, cols);
        const deletedLength = Math.min(length, lineContent.length - prev.length);
        const after = lineContent.substring(cols + deletedLength);
        this.br = false;
        this.content = '';
        this.append(prev + after);
        return deletedLength;
    }

    replace(cols: number, content: string): number {
        const lineContent = this.getContent();
        if (cols >= lineContent.length + this.cols) {
            throw new Error(`${cols} cols is out of range`);
        }
        const prev = lineContent.substring(0, cols);
        const replacedLength = Math.min(content.length, lineContent.length - prev.length);
        const mid = content.substring(0, replacedLength);
        const after = lineContent.substring(cols + replacedLength);
        this.br = false;
        this.content = '';
        this.append(prev + mid + after);
        return replacedLength;
    }

    getContent() {
        return this.content + (this.br ? '\n' : '');
    }

    getFragment(cols: number, length: number): string {
        const lineContent = this.getContent();
        if (cols >= lineContent.length + this.cols) {
            throw new Error(`${cols} cols is out of range`);
        }
        const prev = lineContent.substring(0, cols);
        const fragmentLength = Math.min(length, lineContent.length - prev.length);
        const fragment = lineContent.substring(cols, cols + fragmentLength);
        return fragment;
    }
}

export class EditorBuffer {
    lines: LineBuffer[] = [];

    getContent() {
        let content = '';
        for (const line of this.lines) {
            content += line.getContent();
        }
        return content;
    }

    insertLine(lineNum: number, content: string) {
        let currLineNum = lineNum;
        for (; currLineNum < this.lines.length; currLineNum++) {
            const line = this.lines[currLineNum];
            if (line === undefined) break;
            line.lineNum++;
        }
        const newLine = new LineBuffer(this, lineNum, 0);
        this.lines.splice(lineNum, 0, newLine);
        newLine.append(content);
    }

    appendLine(lineNum: number, cols: number, content: string) {
        const line = this.lines[lineNum];
        if (line == undefined) {
            const newLine = new LineBuffer(this, lineNum, cols);
            this.lines.push(newLine);
            newLine.append(content);
            return;
        }
        line.insert(cols, content);
    }

    delete(startLineNum: number, cols: number, length: number) {
        let lineNum = startLineNum;
        let line = this.lines[lineNum];
        let remainingLen = length;
        do {
            if (line == undefined) {
                break;
            }
            const deletedLen = line.delete(cols, remainingLen);
            if (deletedLen == 0) {
                break;
            }
            remainingLen -= deletedLen;
            lineNum++;
            line = this.lines[lineNum];
            cols = 0;
        } while (remainingLen > 0);
        this.mergeLines(startLineNum);
    }

    mergeLines(startLineNum: number) {
        let lineNum = startLineNum;
        const prevLine = this.lines[lineNum];
        lineNum++;
        for (; lineNum < this.lines.length; lineNum++) {
            const line = this.lines[lineNum];
            if (!prevLine.br) {
                prevLine.append(line.getContent());
                this.lines.splice(lineNum, 1);
                lineNum--;
                continue;
            }
            break;
        }
    }

    replace(startLineNum: number, cols: number, content: string) {
        let lineNum = startLineNum;
        let line = this.lines[lineNum];
        if (line == undefined) {
            this.appendLine(lineNum, 0, new Array(cols).fill(' ').join(''));
            this.appendLine(lineNum, cols, content);
            return;
        }
        let remainingLen = content.length;
        do {
            const replacedLength = line.replace(cols, content);
            remainingLen -= replacedLength;
            content = content.substring(replacedLength);
            lineNum++;
            line = this.lines[lineNum];
            cols = 0;
        } while (remainingLen > 0);
        this.mergeLines(startLineNum);
    }

    getFragment(startLineNum: number, cols: number, length: number): string {
        let lineNum = startLineNum;
        let line = this.lines[lineNum];
        let fragment = '';
        let remainingLen = length;
        do {
            line = this.lines[lineNum];
            if (line == undefined) {
                break;
            }
            const subFragment = line.getFragment(cols, remainingLen);
            lineNum++;
            remainingLen -= subFragment.length;
            fragment += subFragment;
            cols = 0;
        } while (remainingLen > 0);
        return fragment;
    }
}

export interface IPosition {
    line: number;
    col: number;
}

export interface IEditor {
    cursorPos: IPosition;
    editorBuffer: EditorBuffer;
}

export class VisualEditor {
    cursorPos: IPosition = { line: 0, col: 0 };
    editorBuffer: EditorBuffer = new EditorBuffer();
    constructor() { }

    keyPress(event: KeyboardEvent) {
        if (event.key === 'ArrowUp' || event.key.toLowerCase() === 'k') {
            this.cursorPos.line = Math.max(this.cursorPos.line - 1, 0);
        } else if (event.key === 'ArrowDown' || event.key.toLowerCase() === 'j') {
            const maxLine = this.editorBuffer.lines.length - 1;
            this.cursorPos.line = Math.min(this.cursorPos.line + 1, maxLine);
        } else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'h') {
            this.cursorPos.col = Math.max(this.cursorPos.col - 1, 0);
        } else if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'l') {
            const maxCol = this.editorBuffer.lines[this.cursorPos.line].content.length;
            this.cursorPos.col = Math.min(this.cursorPos.col + 1, maxCol);
        }
        const line = this.editorBuffer.lines[this.cursorPos.line];
        const maxCol = line.content.length;
        this.cursorPos.col = Math.min(this.cursorPos.col, maxCol);
    }

    initialize(elem: HTMLElement) {
        const keyboardEvent = fromEvent<KeyboardEvent>(elem, 'keydown');
        return keyboardEvent;
    }

    handleKey(keyboardEvent: Observable<KeyboardEvent>) {
        keyboardEvent
            .pipe(
                buffer(keyboardEvent.pipe(filter((event) => /^[0-9]$/.test(event.key)))),
                map((events) => events.map((event) => event.key)),
                filter((keys) => keys.join('') === 'a'),
            )
            .subscribe(() => {
                console.log("連續輸入數字後跟字母 'a' 觸發");
            });
    }
}

export class NumMoveListener1 {
    _callback: (signal: boolean) => void = () => { };

    constructor(keyboardEvent: Observable<KeyboardEvent>) {
        this.listenEvent(keyboardEvent);
    }

    listenEvent(keyboardEvent: Observable<KeyboardEvent>) {
        keyboardEvent
            .pipe(
                scan((buffer: string[], event) => {
                    buffer.push(event.key);
                    return buffer;
                }, []),
                filter((buffer) => buffer.length >= 2),
                filter((buffer) => /^[a-z]$/.test(buffer[buffer.length - 1])),
            )
            .subscribe(() => {
                this._callback(true);
            });
    }

    isValid(key: string) {
        if (/^[0-9]$/.test(key)) {
            return true;
        }
        if (/^[a-z]$/.test(key)) {
            return true;
        }
        return false;
    }

    listen(callback: (signal: boolean) => void) {
        this._callback = callback;
    }
}

export abstract class InputKeysListener<TData> {
    protected _editor: IEditor = null!;
    private _callback: (data: TData) => void = () => { };
    private _data: TData = null!;
    private _regex: RegExp | null = null;
    private _buffer: string[] = [];

    abstract prefixLength: number;
    abstract regexPattern: string; // = /^\d+[a-zA-Z]\d+$/;
    abstract handle(matches: RegExpExecArray): TData;

    listenEvent(editor: IEditor, keyboardEvent: Observable<KeyboardEvent>) {
        this._editor = editor;
        keyboardEvent
            .pipe(
                scan((buffer: string[], event) => {
                    buffer.push(event.key)
                    return buffer;
                }, []),
                filter((buffer) => buffer.length >= this.prefixLength),
                filter((buffer) => {
                    if (this._regex == null) {
                        this._regex = new RegExp(this.regexPattern);
                    }
                    const input = buffer.join('');
                    const matches = this._regex.exec(input)!;
                    const success = matches != null;
                    if (success) {
                        this._data = this.handle(matches);
                        buffer.length = 0;
                    }
                    return success;
                }),
            )
            .subscribe(() => {
                this._callback(this._data);
            });
    }

    attach(callback: (data: TData) => void) {
        this._callback = callback;
    }
}

export interface INumMoveData {
    lineNum: number;
    suffix: string;
}

export class NumMoveListener extends InputKeysListener<INumMoveData> {
    prefixLength: number = 2;
    regexPattern: string = "^\\d+[a-zA-Z]$";
    handle(matches: RegExpExecArray): INumMoveData {
        return {
            lineNum: parseInt(matches[0]),
            suffix: matches[1],
        }
    }
}


export interface IAzNumAzData {
    prefix: string;
    lineNum: number;
    suffix: string;
}

export class AzNumAzListener extends InputKeysListener<IAzNumAzData> {

    prefixLength: number = 3;
    regexPattern: string = '^\\d+[a-zA-Z]\\d+$';
    handle(matches: RegExpExecArray): IAzNumAzData {
        return {
            prefix: matches[0],
            lineNum: parseInt(matches[1]),
            suffix: matches[2],
        };
    }
}
