import { StringParser } from './StringParser';

export class SqlTokenizer {
    private _parser: StringParser;

    constructor(text: string) {
        this._parser = new StringParser(text);
    }

    nextToken(): string {
        this._parser.skipWhitespace();
        if (this._parser.isEnd()) return '';
        const ch = this._parser.peekNext();
        // 數字
        if (/[0-9]/.test(ch)) {
            const span = this._parser.readInt();
            return span.Word;
        }
        // 單引號或雙引號或反引號
        if (ch === '\'' || ch === '"' || ch === '`') {
            const span = this._parser.readSqlQuotedString();
            return span.Word;
        }
        // 特殊符號
        if (/[^a-zA-Z0-9_\s]/.test(ch)) {
            const span = this._parser.readSymbols();
            return span.Word;
        }
        // 標識符號 (關鍵字、變數等)
        if (/[a-zA-Z_@#$]/.test(ch)) {
            const span = this._parser.readIdentifier();
            return span.Word;
        }
        // 其他情況
        this._parser.nextChar();
        return ch;
    }
}

export function tokenizeSql(sql: string): string[] {
    const tokenizer = new SqlTokenizer(sql);
    const tokens: string[] = [];
    while (true) {
        const token = tokenizer.nextToken();
        if (token === '') break;
        tokens.push(token);
    }
    return tokens;
}