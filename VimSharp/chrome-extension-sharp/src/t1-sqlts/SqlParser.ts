import { StringParser, TextSpan } from "./StringParser";
import { ParseError, SqlType, SqlExpr } from './Expressions/SqlType';

class ParseSqlInput
{
    constructor(text: StringParser) {
        this.text = text;
    }
    text: StringParser;
}

class ParseSqlResult
{
    constructor(text: StringParser, value: ParseError | SqlExpr | null) {
        this.text = text;
        this.error = value instanceof ParseError ? value : null;
        this.value = value instanceof SqlExpr ? value : null;
    }
    text: StringParser;
    error: ParseError | null;
    value: SqlExpr | null;
}

type ParseSqlFn = (input: ParseSqlInput) => ParseSqlResult;

function success(input: ParseSqlInput, value: SqlExpr): ParseSqlResult
{
    return new ParseSqlResult(input.text, value);
}

function none(input: ParseSqlInput): ParseSqlResult
{
    return new ParseSqlResult(input.text, null);
}

function error(input: ParseSqlInput, message: string): ParseSqlResult
{
    return new ParseSqlResult(input.text, new ParseError(message));
}

function skipWhiteSpace(): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        const startPosition = input.text.position;
        while (true)
        {
            const isSkip1 = input.text.skipWhitespace();
            const isSkip2 = input.text.skipSqlComment();
            const isSkip = isSkip1 || isSkip2;
            if (!isSkip)
            {
                break;
            }
        }
        const endPosition = input.text.position;
        return success(input, 
            new SqlExpr(SqlType.TextSpan, 
                new TextSpan(input.text.substring(startPosition, endPosition), startPosition, endPosition-startPosition))
            );
    };
}

function tryKeyword(expected: string): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        skipWhiteSpace()(input);
        const text = input.text;
        const startPosition = text.position;
        if (!text.tryMatch(expected))
        {
            return none(input);
        }
        return new ParseSqlResult(input.text, 
            new SqlExpr(SqlType.TextSpan, 
                new TextSpan(expected, startPosition, text.position - startPosition)));
    };
}

function or(...fns: ParseSqlFn[]): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        const startPosition = input.text.position;
        for (const fn of fns) {
            const result = fn(input);
            if (result.error) {
                result.text.position = startPosition;
                continue;
            }
            return result;
        }
        return none(input);
    }
}

export function parseIntValue(): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        const text = input.text;
        const rc = text.try(() => text.readInt());
        if(rc.success) {
            return success(input, 
                new SqlExpr(SqlType.IntValue, rc.textSpan)
            );
        }
        return error(input, "Expected integer value");
    }
}

function keywords(...keywords: string[]): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        const startPosition = input.text.position;
        let results = [];
        for (const keyword of keywords) {
            const subResult = tryKeyword(keyword)(input);
            if (subResult.error) {
                subResult.text.position = startPosition;
                continue;
            }
            results.push(subResult);
        }
        const endPosition = input.text.position;
        return success(input, 
            new SqlExpr(SqlType.TextSpan, 
                new TextSpan(input.text.substring(startPosition, endPosition), startPosition, endPosition-startPosition))
            );
    }
}

function parseSelectTypeClause(): ParseSqlFn
{
    return (input: ParseSqlInput) => {
        const resultAll = tryKeyword("ALL")(input);
        if(resultAll.value) {
            return new ParseSqlResult(input.text, 
                new SqlExpr(SqlType.AllClause, resultAll.value.Span)
            );
        }
        const resultDistinct = keywords("DISTINCT")(input);
        if (resultDistinct.value) {
            return new ParseSqlResult(input.text, 
                new SqlExpr(SqlType.DistinctClause, resultDistinct.value.Span)
            );
        }
        return new ParseSqlResult(input.text, 
            new SqlExpr(SqlType.NoneClause, TextSpan.None)
        );
    };
}

// function parseTopClause(): ParseSqlFn
// {
//     return (input: ParseSqlInput) => {
//         const result = tryKeyword("TOP")(input);
//         if (result.error) {
//             return result;
//         }

//     }
// }

// function parseSelectStatement(): ParseSqlFn
// {
//     return (input: ParseSqlInput) => {
//         const result = tryKeyword("SELECT")(input);
//         if (result.error) {
//             return result;
//         }
//         const typeClause = parseSelectTypeClause()(input);
//         if (typeClause.error) {
//             return typeClause;
//         }
//     };
// }

export function createParseInput(sql: string): ParseSqlInput
{
    const text = new StringParser(sql);
    return new ParseSqlInput(text);
}

export function parseSql(sql: string): SqlExpr[] {
    const input = createParseInput(sql);
    const result = parseIntValue()(input);
    return [result.value!];
}
