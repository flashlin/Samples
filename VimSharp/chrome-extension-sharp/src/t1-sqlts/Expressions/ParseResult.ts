import { ParseError } from './ParseError';
import { IParseResult } from './IParseResult';

export class ParseResult<T> implements IParseResult {
    result?: T;
    hasResult: boolean = false;
    error: ParseError = ParseError.Empty;
    hasError: boolean = false;

    constructor(result?: T);
    constructor(error: ParseError);
    constructor(param?: T | ParseError) {
        if (param instanceof ParseError) {
            this.hasError = true;
            this.error = param;
        } else if (param !== undefined) {
            this.hasResult = true;
            this.result = param;
        }
    }

    get object(): any {
        return this.result;
    }

    get objectValue(): any {
        if (this.result == null) throw new Error('Result is null');
        return this.result;
    }

    get hasValue(): boolean {
        return this.hasResult && this.result != null;
    }

    get resultValue(): T {
        if (this.result == null) throw new Error('Result is null');
        return this.result;
    }

    /**
     * 將當前的 ParseResult<T> 轉換為 ParseResult<T1>
     * 相當於 C# 版本的 To<T1>() 方法
     */
    To<T1>(): ParseResult<T1> {
        if (this.hasError) {
            return ParseResult.Error<T1>(this.error);
        }
        return new ParseResult<T1>(this.result as unknown as T1);
    }

    static Error<T>(error: ParseError): ParseResult<T> {
        return new ParseResult<T>(error);
    }

    /**
     * 從結果值創建 ParseResult<T>
     * 相當於 C# 版本的 implicit operator ParseResult<T>(T? result)
     */
    static From<T>(result?: T): ParseResult<T> {
        return new ParseResult<T>(result);
    }
} 