export class TextSpan {
    static None: TextSpan = new TextSpan();
    Word: string = '';
    Offset: number = -1;
    Length: number = 0;

    constructor(word: string = '', offset: number = -1, length: number = 0) {
        this.Word = word;
        this.Offset = offset;
        this.Length = length;
    }

    static Empty(startPosition: number): TextSpan {
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = 0;
        return span;
    }
}

export class StringParser {
    private _text: string;
    private _position: number = 0;
    private _previousWord: TextSpan = new TextSpan();
    private static readonly Brackets: string[] = ['(', ')', '{', '}', '[', ']'];

    constructor(text: string) {
        this._text = text;
    }

    get position(): number {
        return this._position;
    }
    set position(value: number) {
        this._position = value;
    }

    getRemainingText(): string {
        if (this.isEnd()) return '';
        return this.substring(this._position);
    }

    getPreviousText(offset: number): string {
        return this.substring(offset, this._position);
    }

    getText(startOffset: number, endOffset: number): string {
        return this.substring(startOffset, endOffset - startOffset);
    }

    isEnd(): boolean {
        return this._position >= this._text.length;
    }

    isPeekIgnoreCase(readFn: () => TextSpan, expected: string): boolean {
        const startPosition = this._position;
        const textSpan = readFn();
        this._position = startPosition;
        return StringParser.isSameIgnoreCase(textSpan.Word, expected);
    }

    isWordChar(c: string): boolean {
        return /[a-zA-Z0-9_@#$]/.test(c);
    }

    matchSymbol(expected: string): TextSpan {
        this.skipWhitespace();
        const startPosition = this._position;
        for (const c of expected) {
            if (this.isEnd()) throw new Error(`Expected '${expected}' at position ${this._position}, but found EOF.`);
            const nextChar = this.nextChar();
            if (nextChar !== c) throw new Error(`Expected '${expected}' at position ${this._position}, but found different content '${nextChar}'.`);
        }
        const span = new TextSpan();
        span.Word = this.substring(startPosition, this._position);
        span.Offset = startPosition;
        span.Length = this._position - startPosition;
        return span;
    }

    nextChar(): string {
        if (this.isEnd()) return '\0';
        return this._text[this._position++];
    }

    peek(readFunc: () => TextSpan): TextSpan {
        const tempPosition = this._position;
        const textSpan = readFunc();
        this._position = tempPosition;
        return textSpan;
    }

    peekChar(): string {
        this.skipWhitespace();
        if (this.isEnd()) return '\0';
        return this._text[this._position];
    }

    peekIdentifier(word: string): TextSpan {
        const span = this.peek(() => this.readIdentifier());
        if (span.Word === word) return span;
        const spanEmpty = new TextSpan();
        spanEmpty.Word = '';
        spanEmpty.Offset = this._position;
        spanEmpty.Length = 0;
        return spanEmpty;
    }

    peekMatchSymbol(symbol: string): boolean {
        const tempPosition = this._position;
        const isSymbol = this.nextText(symbol.length).Word === symbol;
        this._position = tempPosition;
        return isSymbol;
    }

    peekNext(): string {
        if (this.isEnd()) return '\0';
        return this._text[this._position];
    }

    peekWord(): TextSpan {
        this.skipWhitespace();
        let tempPosition = this._position;
        while (tempPosition < this._text.length && this.isWordChar(this._text[tempPosition])) {
            tempPosition++;
        }
        const span = new TextSpan();
        span.Word = this.substring(this._position, tempPosition);
        span.Offset = this._position;
        span.Length = tempPosition - this._position;
        return span;
    }

    previousWord(): TextSpan {
        return this._previousWord;
    }

    readChar(): string {
        this.skipWhitespace();
        if (this.isEnd()) return '\0';
        return this._text[this._position++];
    }

    readDoubleComment(): TextSpan {
        const startPosition = this._position;
        const { success, textSpan: openSymbol } = this.try(() => this.readSymbols());
        if (success && openSymbol.Word === '/*') {
            this.readUntil('*/');
            this.nextText(2);
            const span = new TextSpan();
            span.Word = this.substring(startPosition, this._position);
            span.Offset = startPosition;
            span.Length = this._position - startPosition;
            return span;
        }
        this._position = startPosition;
        const span = new TextSpan();
        span.Word = '';
        span.Offset = startPosition;
        span.Length = 0;
        return span;
    }

    readFloat(): TextSpan {
        this.skipWhitespace();
        const startOffset = this._position;
        const { success } = this.try(() => this.readInt());
        if (!success) return new TextSpan('', this._position, 0);
        const dot = this.nextChar();
        if (dot !== '.') {
            this._position = startOffset;
            return new TextSpan('', this._position, 0);
        }
        this.readInt();
        const span = new TextSpan();
        span.Word = this.substring(startOffset, this._position);
        span.Offset = startOffset;
        span.Length = this._position - startOffset;
        return span;
    }

    readFullQuotedIdentifier(): TextSpan {
        this.skipWhitespace();
        const startPosition = this._position;
        let prevToken: TextSpan = TextSpan.None;
        while (!this.isEnd()) {
            const identifier = this.or(() => this.readIdentifier(), () => this.readQuotedIdentifier());
            if (identifier.Length === 0) {
                if (prevToken === TextSpan.None) return new TextSpan('', startPosition, 0);
                break;
            }
            prevToken = identifier;
            const dotdot = this.peek(() => this.nextText(2));
            if (dotdot.Word === '..') {
                this.nextText(2);
                prevToken = new TextSpan(dotdot.Word, dotdot.Offset, 2);
                continue;
            }
            if (dotdot.Word === '.*') {
                this.nextText(2);
                break;
            }
            if (this.peekNext() !== '.') break;
            const dot = this.nextChar();
            prevToken = new TextSpan(dot, this._position - 1, 1);
        }
        let lastPosition = this._position;
        if (prevToken.Word === '.') lastPosition = prevToken.Offset;
        const span = new TextSpan();
        span.Word = this.substring(startPosition, lastPosition);
        span.Offset = startPosition;
        span.Length = lastPosition - startPosition;
        return span;
    }

    readIdentifier(): TextSpan {
        this.skipWhitespace();
        const offset = this._position;
        const ch = this.peekChar();
        if (!/[a-zA-Z_@#$]/.test(ch)) {
            const span = new TextSpan();
            span.Word = '';
            span.Offset = this._position;
            span.Length = 0;
            return span;
        }
        while (!this.isEnd()) {
            const c = this.nextChar();
            if (!this.isWordChar(c)) {
                this._position--;
                break;
            }
        }
        const identifyPrev = ['@', '#', '$'];
        const identifier = this.substring(offset, this._position);
        if (identifyPrev.includes(identifier)) {
            const span = new TextSpan();
            span.Word = '';
            span.Offset = offset;
            span.Length = 0;
            return span;
        }
        const span = new TextSpan();
        span.Word = identifier;
        span.Offset = offset;
        span.Length = this._position - offset;
        return span;
    }

    readInt(): TextSpan {
        this.skipWhitespace();
        const startOffset = this._position;
        let ch = this.peekChar();
        if (!/[0-9]/.test(ch)) {
            const span = new TextSpan();
            span.Word = '';
            span.Offset = this._position;
            span.Length = 0;
            return span;
        }
        while (!this.isEnd()) {
            ch = this.nextChar();
            if (!/[0-9]/.test(ch)) {
                this._position--;
                break;
            }
        }
        const span = new TextSpan();
        span.Word = this.substring(startOffset, this._position);
        span.Offset = startOffset;
        span.Length = this._position - startOffset;
        return span;
    }

    readNegativeNumber(): TextSpan {
        this.skipWhitespace();
        if (this.peekNext() !== '-') {
            const span = new TextSpan();
            span.Word = '';
            span.Offset = this._position;
            span.Length = 0;
            return span;
        }
        const startPosition = this._position;
        this.nextChar();
        const floatNumber = this.readFloat();
        if (floatNumber.Length !== 0) {
            return new TextSpan(this.substring(startPosition, this._position), startPosition, 0);
        }
        this.readInt();
        return new TextSpan(this.substring(startPosition, this._position), startPosition, this._position - startPosition);
    }

    readQuotedIdentifier(): TextSpan {
        let quoteChar = this.peekChar();
        if (quoteChar !== '"' && quoteChar !== '[' && quoteChar !== '`') {
            const span = new TextSpan();
            span.Word = '';
            span.Offset = this._position;
            span.Length = 0;
            return span;
        }
        const offset = this._position;
        const closeChar = quoteChar === '[' ? ']' : quoteChar;
        let identifier = quoteChar;
        this.readChar();
        while (!this.isEnd()) {
            const c = this.nextChar();
            identifier += c;
            if (c === closeChar) break;
        }
        const span = new TextSpan();
        span.Word = identifier;
        span.Offset = offset;
        span.Length = this._position - offset;
        return span;
    }

    readSqlDate(): TextSpan {
        const startPosition = this._position;
        const year = this.readInt();
        this.nextChar();
        const month = this.readInt();
        this.nextChar();
        const day = this.readInt();
        if (year.Length === 0 || month.Length === 0 || day.Length === 0) {
            this._position = startPosition;
            const span = new TextSpan();
            span.Word = '';
            span.Offset = startPosition;
            span.Length = 0;
            return span;
        }
        const span = new TextSpan();
        span.Word = this.substring(startPosition, this._position);
        span.Offset = startPosition;
        span.Length = this._position - startPosition;
        return span;
    }

    readSqlIdentifier(): TextSpan {
        let result = this.try(() => this.readFullQuotedIdentifier());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readIdentifier());
        if (result.success) return result.textSpan;
        return new TextSpan('', this._position, 0);
    }

    readSqlQuotedString(): TextSpan {
        let quoteChar = this.peekChar();
        if (quoteChar !== '\'' && quoteChar !== '"' && quoteChar !== '`' && quoteChar !== 'N') return new TextSpan('', this._position, 0);
        const startPosition = this._position;
        const startChar = this.readChar();
        if (startChar === 'N') {
            quoteChar = this.nextChar();
            if (quoteChar !== '\'' && quoteChar !== '"' && quoteChar !== '`') {
                this._position = startPosition;
                return new TextSpan('', startPosition, 0);
            }
        }
        while (!this.isEnd()) {
            const c = this.nextChar();
            if (c === quoteChar && this.peekNext() === quoteChar) {
                this.nextChar();
                continue;
            }
            if (c === quoteChar) break;
        }
        const span = new TextSpan();
        span.Word = this.substring(startPosition, this._position);
        span.Offset = startPosition;
        span.Length = this._position - startPosition;
        return span;
    }

    readSqlSingleComment(): TextSpan {
        const startPosition = this._position;
        this.readChar();
        this.nextChar();
        this.readUntil((c) => c === '\n');
        this.nextChar();
        const span = new TextSpan();
        span.Word = this.substring(startPosition, this._position);
        span.Offset = startPosition;
        span.Length = this._position - startPosition;
        return span;
    }

    nextText(length: number): TextSpan {
        length = Math.min(length, this._text.length - this._position);
        const span = new TextSpan();
        span.Word = this.substring(this._position, this._position + length);
        span.Offset = this._position;
        span.Length = length;
        this._position += length;
        return span;
    }

    readBracket(): TextSpan {
        this.skipWhitespace();
        if (this.isEnd()) return new TextSpan('', this._position, 0);
        const startPosition = this._position;
        const bracketStr = this.nextText(1).Word;
        if (!StringParser.Brackets.includes(bracketStr[0])) return new TextSpan('', startPosition, 0);
        const span = new TextSpan();
        span.Word = bracketStr;
        span.Offset = startPosition;
        span.Length = 1;
        return span;
    }

    readSymbol(length: number): TextSpan {
        this.skipWhitespace();
        const startPosition = this._position;
        const symbol = this.nextText(length);
        return symbol;
    }

    private isSymbolEnd(symbolEnd: string): boolean {
        if (symbolEnd === '\0') return true;
        if (this.isWordChar(symbolEnd)) return true;
        if (/\s/.test(symbolEnd)) return true;
        if (symbolEnd === '\'') return true;
        return StringParser.Brackets.includes(symbolEnd);
    }

    readSymbols(): TextSpan {
        this.skipWhitespace();
        const offset = this._position;
        const ch = this.peekChar();
        if (/[a-zA-Z]/.test(ch) && ch !== '_') return new TextSpan('', this._position, 0);
        let symbol = '';
        while (!this.isEnd()) {
            const c = this.nextChar();
            if (this.isWordChar(c) || /\s/.test(c)) {
                this._position--;
                break;
            }
            symbol += c;
        }
        const span = new TextSpan();
        span.Word = symbol;
        span.Offset = offset;
        span.Length = this._position - offset;
        return span;
    }

    readUntil(predicate: ((c: string) => boolean) | string): TextSpan {
        const offset = this._position;
        if (typeof predicate === 'string') {
            while (!this.isEnd() && this.peekString(predicate.length).join('') !== predicate) {
                this.nextChar();
            }
        } else {
            while (!this.isEnd() && !predicate(this.peekNext())) {
                this.nextChar();
            }
        }
        const span = new TextSpan();
        span.Word = this.substring(offset, this._position);
        span.Offset = offset;
        span.Length = this._position - offset;
        return span;
    }

    readUntilRightParenthesis(): TextSpan {
        const startPosition = this._position;
        let openParenthesis = 0;
        while (!this.isEnd()) {
            const c = this.nextChar();
            if (c === '(') {
                openParenthesis++;
                continue;
            }
            if (c === ')') {
                openParenthesis--;
                if (openParenthesis === -1) {
                    this._position--;
                    const span = new TextSpan();
                    span.Word = this.substring(startPosition, this._position);
                    span.Offset = startPosition;
                    span.Length = this._position - startPosition;
                    return span;
                }
            }
        }
        this._position = startPosition;
        return new TextSpan('', startPosition, 0);
    }

    skipSqlComment(): boolean {
        const isSkipSqlDoubleComment = this.skipSqlDoubleComment();
        const isSkipSqlSingleComment = this.skipSqlSingleComment();
        return isSkipSqlDoubleComment || isSkipSqlSingleComment;
    }

    skipSqlDoubleComment(): boolean {
        const startPosition = this._position;
        const { success, textSpan: openSymbol } = this.try(() => this.readSymbols());
        if (success && openSymbol.Word === '/**/') return true;
        if (success && openSymbol.Word === '/*') {
            this._position = startPosition;
            this.readDoubleComment();
            return true;
        }
        this._position = startPosition;
        return false;
    }

    skipSqlSingleComment(): boolean {
        const startPosition = this._position;
        const { success, textSpan: openSymbol } = this.try(() => this.readSymbols());
        if (success && openSymbol.Word.startsWith('--')) {
            this._position = startPosition;
            this.readSqlSingleComment();
            return true;
        }
        this._position = startPosition;
        return false;
    }

    skipWhitespace(): boolean {
        let isSkip = false;
        while (!this.isEnd() && /\s/.test(this._text[this._position])) {
            this._position++;
            isSkip = true;
        }
        return isSkip;
    }

    try(readFunc: () => TextSpan): { success: boolean, textSpan: TextSpan } {
        const startPosition = this._position;
        const textSpan = readFunc();
        if (textSpan.Length === 0) {
            this._position = startPosition;
            return { success: false, textSpan };
        }
        this._previousWord = textSpan;
        return { success: true, textSpan };
    }

    tryTextIgnoreCase(text: string): { success: boolean, textSpan: TextSpan } {
        this.skipWhitespace();
        const startPosition = this._position;
        const result = this.try(() => this.nextText(text.length));
        if (result.success && result.textSpan.Word.toLowerCase() === text.toLowerCase()) {
            return { success: true, textSpan: result.textSpan };
        }
        this._position = startPosition;
        return { success: false, textSpan: new TextSpan('', startPosition, 0) };
    }

    tryMatch(symbol: string): { success: boolean, textSpan: TextSpan } {
        this.skipWhitespace();
        const startPosition = this._position;
        const result = this.try(() => this.nextText(symbol.length));
        if (result.success && result.textSpan.Word === symbol) {
            return { success: true, textSpan: result.textSpan };
        }
        this._position = startPosition;
        return { success: false, textSpan: new TextSpan('', startPosition, 0) };
    }

    tryMatches(...keywords: string[]): boolean {
        this.skipWhitespace();
        const startPosition = this._position;
        for (const keyword of keywords) {
            if (!this.tryMatch(keyword).success) {
                this._position = startPosition;
                return false;
            }
        }
        return true;
    }

    tryKeywordsIgnoreCase(keywords: string[]): { success: boolean, textSpan: TextSpan } {
        this.skipWhitespace();
        const startPosition = this._position;
        for (const keyword of keywords) {
            if (!this.tryKeywordIgnoreCase(keyword).success) {
                this._position = startPosition;
                return { success: false, textSpan: TextSpan.Empty(startPosition) };
            }
        }
        const span = new TextSpan();
        span.Word = this.substring(startPosition, this._position);
        span.Offset = startPosition;
        span.Length = this._position - startPosition;
        return { success: true, textSpan: span };
    }

    tryKeywordIgnoreCase(keyword: string): { success: boolean, textSpan: TextSpan } {
        this.skipWhitespace();
        const startPosition = this._position;
        let readCount = 0;
        while (!this.isEnd() && readCount < keyword.length) {
            readCount++;
            this._position++;
        }
        const word = this.substring(startPosition, startPosition + readCount);
        if (word.toLowerCase() !== keyword.toLowerCase()) {
            this._position = startPosition;
            return { success: false, textSpan: TextSpan.Empty(startPosition) };
        }
        const nextChar = this.peekNext();
        if (this.isWordChar(nextChar)) {
            this._position = startPosition;
            return { success: false, textSpan: TextSpan.Empty(startPosition) };
        }
        this._previousWord = new TextSpan(word, this._position, keyword.length);
        const span = new TextSpan();
        span.Word = word;
        span.Offset = this._position - keyword.length;
        span.Length = keyword.length;
        return { success: true, textSpan: span };
    }

    private static isSameIgnoreCase(word1: string, word2: string): boolean {
        return word1.toLowerCase() === word2.toLowerCase();
    }

    private or(...readFnList: Array<() => TextSpan>): TextSpan {
        for (const readFn of readFnList) {
            const textSpan = readFn();
            if (textSpan.Length !== 0) return textSpan;
        }
        return new TextSpan('', this._position, 0);
    }

    private peekString(length: number): string[] {
        if (this.isEnd()) return [];
        const remainLength = this._text.length - this._position;
        const readLength = Math.min(length, remainLength);
        return this._text.substr(this._position, readLength).split('');
    }

    createEmptySpan(): TextSpan {
        return new TextSpan('', this._position, 0);
    }

    createSpan(startSpan: TextSpan): TextSpan {
        const span = new TextSpan();
        span.Word = this.substring(startSpan.Offset, this._position);
        span.Offset = startSpan.Offset;
        span.Length = this._position - startSpan.Offset;
        return span;
    }

    createSpan2(startSpan: TextSpan, endSpan: TextSpan): TextSpan {
        const length = endSpan.Offset + endSpan.Length - startSpan.Offset;
        const span = new TextSpan();
        span.Word = this.substring(startSpan.Offset, startSpan.Offset + length);
        span.Offset = startSpan.Offset;
        span.Length = length;
        return span;
    }

    createSpanFromOffset(startOffset: number): TextSpan {
        const span = new TextSpan();
        span.Word = this.substring(startOffset, this._position);
        span.Offset = startOffset;
        span.Length = this._position - startOffset;
        return span;
    }

    readNextSqlToken(): TextSpan {
        this.skipWhitespace();
        this.skipSqlComment();
        const startPosition = this._position;
        if (this.isEnd()) return new TextSpan('', startPosition, 0);
        let result = this.try(() => this.readSqlIdentifier());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readSqlQuotedString());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readSqlDate());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readQuotedIdentifier());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readIdentifier());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readFloat());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readInt());
        if (result.success) return result.textSpan;
        result = this.try(() => this.readSymbols());
        if (result.success) return result.textSpan;
        const span = this.readUntil((c) => c === '\n');
        return span;
    }

    substring(start: number, end?: number): string {
        return this._text.substring(start, end);
    }
} 