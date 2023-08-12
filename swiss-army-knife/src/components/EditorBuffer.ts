export class LineBuffer {
    prev: LineBuffer | null = null;
    next: LineBuffer | null = null;
    lineNum = 0;
    cols = 0;
    tabWidth = 3;
    placeholder = true;
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
        if (content == undefined || content === '') {
            this.content = '';
            this.br = false;
            this.placeholder = true;
            return;
        }

        let line = '';
        let n = 0;
        for (; n < content.length; n++) {
            const c = content[n];
            if (c === undefined) break;
            if (c == '\n') {
                n++;
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
}

export class EditorBuffer {
    lines: LineBuffer[] = [];

    getContent() {
        let content = '';
        for (const line of this.lines) {
            content += line.content
        }
        return content
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
        const line = new LineBuffer(this, lineNum, cols);
        line.append(content);
        this.lines.push(line);
    }
}
