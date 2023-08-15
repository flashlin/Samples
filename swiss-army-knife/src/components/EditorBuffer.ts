import { Observable, Subject, fromEvent } from 'rxjs';
import { filter, map, scan, takeUntil } from 'rxjs/operators';

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

export interface IArea {
    startLineNum: number;
    startCol: number;
    width: number;
    height: number;
}

export interface IEditor {
    cursorPos: IPosition;
    editorBuffer: EditorBuffer;
    area: IArea;
}

export class VisualEditor {
    cursorPos: IPosition = { line: 0, col: 0 };
    area: IArea = {
        startLineNum: 0,
        startCol: 0,
        width: 80,
        height: 40,
    };
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
        this.rearrangePos();
    }

    rearrangePos() {
        const line = this.editorBuffer.lines[this.cursorPos.line];
        const maxCol = line.content.length;
        this.cursorPos.col = Math.min(this.cursorPos.col, maxCol);
    }

    initialize(elem: HTMLElement) {
        const keyboardEvent = fromEvent<KeyboardEvent>(elem, 'keyup');
        this.handleKey(keyboardEvent);
    }

    handleKey(keyboardEvent: Observable<KeyboardEvent>) {
        const t1 = new MoveListener();
        t1.listenEvent(this, keyboardEvent);
    }
}

export class EditorRender {
    fontSize = 24;
    fontName = '文泉驛等寬微米黑';
    fontWidth = 0;
    fontHeight = 0;
    editor: IEditor;
    graph: CanvasRenderingContext2D = null!;

    constructor(editor: IEditor) {
        this.editor = editor;
        this.initFontWH();
    }

    initFontWH() {
        const t = document.createElement('span');
        t.style.fontFamily = this.fontName;
        t.style.fontSize = `${this.fontSize}`;
        t.textContent = 'y';
        document.body.appendChild(t);
        this.fontWidth = t.offsetWidth + 2;
        this.fontHeight = t.offsetHeight + 2;
        document.body.removeChild(t);
    }

    createCanvas(elem: HTMLElement) {
        const editor = this.editor;
        const canvas = document.createElement('canvas');
        //canvas.id = this.id;
        canvas.width = editor.area.width * this.fontWidth + 4;
        canvas.height = editor.area.height * this.fontHeight + 4;
        canvas.style.cursor = 'text';
        canvas.style.border = 'solid #0f0 1px';
        //document.body.appendChild(canvas);
        elem.appendChild(canvas);

        const graph = canvas.getContext('2d')!;
        graph.textAlign = 'left';
        this.graph = graph;
    }

    showCursorAt(x: number, y: number, show_cursor: boolean) {
        //const editor = this.editor;
        // const clr = this.buffer.getColorAt(x, y),
        //     chr = this.buffer.getCharAt(x, y);
        const clr = 0xff0000;
        const chr = 0xffffff;
        // if (show_cursor &&
        //     x == editor.buffer.x &&
        //     y == this.buffer.y)
        //     clr = ((clr >> 4) | (clr << 4)) & 0xFF;
        this.drawRect(x, y, clr);
        this.drawChar(x, y, clr & 0x0f, chr);
    }

    drawRect(x: number, y: number, color: number) {
        const c = this.graph;
        //c.fillStyle = color;
        c.fillRect(x * this.fontWidth + 2, y * this.fontHeight + 4, this.fontWidth, this.fontHeight);
    }

    drawChar(x: number, y: number, clr: number, chr: number) {
        const c = this.graph;
        c.fillStyle = `${clr}`;
        //c.font = (clr.h ? 'bold ' : '') + this.fontSize + 'px ' + this.fontName;
        c.fillText(`${chr}`, x * this.fontWidth + 2, (y + 1) * this.fontHeight);
    }
}

export abstract class InputKeysListener<TData> {
    protected _editor: IEditor = null!;
    private _regex: RegExp | null = null;
    private _callback: (data: TData) => void = () => { };

    abstract prefixLength: number;
    abstract regexPattern: string; // = /^\d+[a-zA-Z]\d+$/;
    abstract toData(matches: RegExpExecArray): TData;
    abstract handle(data: TData): void;

    listenEvent(editor: IEditor, keyboardEvent: Observable<KeyboardEvent>) {
        this._editor = editor;
        const unsubscribe$ = new Subject();
        keyboardEvent
            .pipe(
                scan((buffer: string[], event) => {
                    buffer.push(event.key);
                    return buffer;
                }, []),
                filter((buffer) => buffer.length >= this.prefixLength),
                map((buffer) => {
                    if (this._regex == null) {
                        this._regex = new RegExp(this.regexPattern);
                    }
                    const input = buffer.join('');
                    const matches = this._regex.exec(input);
                    if (matches != null) {
                        buffer.length = 0;
                    }
                    return matches;
                }),
                filter((matches) => matches != null),
            )
            .pipe(takeUntil(unsubscribe$))
            .subscribe((matches) => {
                const data = this.toData(matches!);
                this._callback(data);
                unsubscribe$.next(undefined);
                unsubscribe$.complete();
            });
    }

    attach(callback: (data: TData) => void) {
        this._callback = callback;
    }
}

export interface IMoveData {
    key: string;
}

export class MoveListener extends InputKeysListener<IMoveData> {
    prefixLength: number = 1;
    regexPattern: string = '^([jklh])$';
    toData(matches: RegExpExecArray): IMoveData {
        return {
            key: matches[1],
        };
    }
    handle(event: IMoveData): void {
        const editor = this._editor;
        if (event.key === 'ArrowUp' || event.key.toLowerCase() === 'k') {
            editor.cursorPos.line = Math.max(editor.cursorPos.line - 1, 0);
        } else if (event.key === 'ArrowDown' || event.key.toLowerCase() === 'j') {
            const maxLine = editor.editorBuffer.lines.length - 1;
            editor.cursorPos.line = Math.min(editor.cursorPos.line + 1, maxLine);
        } else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'h') {
            editor.cursorPos.col = Math.max(editor.cursorPos.col - 1, 0);
        } else if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'l') {
            const maxCol = editor.editorBuffer.lines[editor.cursorPos.line].content.length;
            editor.cursorPos.col = Math.min(editor.cursorPos.col + 1, maxCol);
        }
        const line = editor.editorBuffer.lines[editor.cursorPos.line];
        const maxCol = line.content.length;
        editor.cursorPos.col = Math.min(editor.cursorPos.col, maxCol);
    }
}

export interface INumMoveData {
    lineNum: number;
    suffix: string;
}

export class NumMoveListener extends InputKeysListener<INumMoveData> {
    prefixLength: number = 2;
    regexPattern: string = '^(\\d+)([a-zA-Z])$';
    toData(matches: RegExpExecArray): INumMoveData {
        return {
            lineNum: parseInt(matches[1]),
            suffix: matches[2],
        };
    }
    handle(data: INumMoveData): void { }
}

export interface IAzNumAzData {
    prefix: string;
    lineNum: number;
    suffix: string;
}

export class AzNumAzListener extends InputKeysListener<IAzNumAzData> {
    prefixLength: number = 3;
    regexPattern: string = '^\\d+[a-zA-Z]\\d+$';
    toData(matches: RegExpExecArray): IAzNumAzData {
        return {
            prefix: matches[0],
            lineNum: parseInt(matches[1]),
            suffix: matches[2],
        };
    }
    handle(data: IAzNumAzData): void { }
}
