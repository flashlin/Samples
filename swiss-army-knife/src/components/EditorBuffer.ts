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
}
