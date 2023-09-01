//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, Lexer, EmbeddedActionsParser } from "chevrotain";

const RULES = {
    AND: "AND",
    GREATER_THAN: ">",
    LESS_THAN: "<",
    selectExpression: "selectExpression",
    aliaName: "aliaName",
    sourceClause: "sourceClause",
    tableClause: "tableClause",
    databaseTableClause: "databaseTableClause",
    columnsExpr: "columnsExpr",
    tableExpr: "tableExpr",
    tableFieldExpr: "tableFieldExpr",
    parenthesisExpr: "parenthesisExpr",
    columnList: "columnList",
    newColumns: "newColumns",
    columnEqualExpr: "columnEqual",
    compareOper: "compareOper",
    extractAtomExpr: "extractAtomExpr",
    extractExpressions: "extractExpressions",
    extractCompareExprs: "extractCompareExprs",
    extractCompareExpr: "extractCompareExpr",
    extractAndOrExprs: "extractAndOrExprs",
    extractAndExpr: "extractAndExpr",
    extractOrExpr: "extractOrExpr",
    integer: "integer",
    integerExpr: "integerExpr",
    expr: "expr",
    fetchColumn: "fetchColumn",
    whereExpr: "whereExpr",
    Identifier: "identifier",
    IDENTIFIER: "Identifier",
    FLOAT: "Float",
    INTEGER: "Integer",
    StringDoubleQuote: "StringDoubleQuote",
    StringSimpleQuote: "StringSimpleQuote",
    SELECT: "SELECT",
    FROM: "FROM",
    WHERE: "WHERE",
    IN: "IN",
    OR: "OR",
    NOT: "NOT",
    NEW: "NEW",
    DOT: "DOT",
    ASSIGN: "ASSIGN",
    LEFT_BRACE: "LEFT_BRACE",
    RIGHT_BRACE: "RIGHT_BRACE",
    COMMA: "COMMA",
    EQUAL: "==",
    GREATER_EQUAL: ">=",
    LESS_EQUAL: "<=",
    NOT_EQUAL: "!=",
    LPAREN: "(",
    RPAREN: ")",

    // special
    TAKE: "TAKE",
    takeExpr: "takeExpr",
};

const StringDoubleQuote = createToken({ name: RULES.StringDoubleQuote, pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: RULES.StringSimpleQuote, pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const Float = createToken({ name: RULES.FLOAT, pattern: /\d+\.\d*/ });
const Integer = createToken({ name: RULES.INTEGER, pattern: /\d+/ });
const SELECT = createToken({ name: RULES.SELECT, pattern: /select/ });
const FROM = createToken({ name: RULES.FROM, pattern: /from/ });
const WHERE = createToken({ name: RULES.WHERE, pattern: /where/ });
const IN = createToken({ name: RULES.IN, pattern: /in/ });
const AND = createToken({ name: RULES.AND, pattern: /(&&)/ });
const OR = createToken({ name: RULES.OR, pattern: /(\|\|)/ });
const NOT = createToken({ name: RULES.NOT, pattern: /(\!)/ });
const NEW = createToken({ name: RULES.NEW, pattern: /new/ });
const DOT = createToken({ name: RULES.DOT, pattern: /(\.)/ });
const ASSIGN = createToken({ name: RULES.ASSIGN, pattern: /(\=)/ });
const LBRACE = createToken({ name: RULES.LEFT_BRACE, pattern: /\{/ });
const RBRACE = createToken({ name: RULES.RIGHT_BRACE, pattern: /\}/ });
const COMMA = createToken({ name: RULES.COMMA, pattern: /\,/ });
const GREATER_THAN = createToken({ name: RULES.GREATER_THAN, pattern: /\>/ });
const LESS_THAN = createToken({ name: RULES.LESS_THAN, pattern: /\</ });
const EQUAL = createToken({ name: RULES.EQUAL, pattern: /\=\=/ });
const GREATER_EQUAL = createToken({ name: RULES.GREATER_EQUAL, pattern: /\>\=/ });
const LESS_EQUAL = createToken({ name: RULES.LESS_EQUAL, pattern: /\<\=/ });
const NOT_EQUAL = createToken({ name: RULES.NOT_EQUAL, pattern: /\!\=/ });
const LPAREN = createToken({ name: RULES.LPAREN, pattern: /\(/ });
const RPAREN = createToken({ name: RULES.RPAREN, pattern: /\)/ });

const TAKE = createToken({ name: RULES.TAKE, pattern: /take/ });

const WhiteSpace = createToken({
    name: "WhiteSpace",
    pattern: /[ \t\n\r]+/,
    group: Lexer.SKIPPED
});

const allTokens = [
    WhiteSpace,
    SELECT,
    WHERE,
    Float,
    Integer,
    FROM,
    AND,
    NOT,
    NEW,
    IN,
    OR,
    DOT,
    EQUAL,
    GREATER_EQUAL,
    LESS_EQUAL,
    NOT_EQUAL,
    LBRACE,
    RBRACE,
    LPAREN,
    RPAREN,
    COMMA,
    ASSIGN,
    GREATER_THAN,
    LESS_THAN,
    StringDoubleQuote,
    StringSimpleQuote,
    //special
    TAKE,
    Identifier,
];


export interface ISqlExpression {
    type: string;
}

export interface ITableFieldExpression extends ISqlExpression {
    aliasTableName: string;
    field: string;
    aliasFieldName: string;
}

export interface ITableExpression extends ISqlExpression {
    name: string;
}

export interface ISelectExpression extends ISqlExpression {
    source: ISqlExpression;
    aliasTableName: string;
    columns: ISelectExpression[];
    where: ISelectExpression | undefined;
}

const linqLexer = new Lexer(allTokens);
class LinqParserEmbedded extends EmbeddedActionsParser {
    constructor() {
        super(allTokens);
        this.performSelfAnalysis();
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        this.CONSUME(FROM);
        const aliasName = this.SUBRULE(this.aliaName);
        this.CONSUME(IN);
        const source = this.SUBRULE(this.sourceClause);
        let where;
        this.OPTION(() => {
            where = this.SUBRULE(this.whereExpr);
        });
        this.CONSUME(SELECT);
        const columns = this.SUBRULE(this.columnsExpr);
        return {
            type: "SELECT_CLAUSE",
            source: source,
            aliasTableName: aliasName,
            columns: columns,
            where: where
        } as ISelectExpression;
    });

    //special
    public takeExpr = this.RULE(RULES.takeExpr, () => {
        this.CONSUME(TAKE);
        const count = this.SUBRULE(this.integer);
        return {
            type: "TOP",
            count: count
        };
    });

    public aliaName = this.RULE(RULES.aliaName, () => {
        return this.SUBRULE(this.identifier);
    });

    public sourceClause = this.RULE(RULES.sourceClause, () => {
        return this.OR([
            { ALT: () => this.SUBRULE(this.databaseTableClause) },
            { ALT: () => this.SUBRULE(this.tableClause) },
        ]);
    });

    public tableClause = this.RULE(RULES.tableClause, () => {
        const name = this.SUBRULE(this.identifier);
        return {
            type: 'TABLE_CLAUSE',
            name: name,
        } as ITableExpression;
    });

    public databaseTableClause = this.RULE(RULES.databaseTableClause, () => {
        const dbName = this.CONSUME(Identifier).image;
        this.CONSUME2(DOT);
        const tbName = this.CONSUME3(Identifier).image;
        return {
            type: 'DATABASE_TABLE_CLAUSE',
            databaseName: dbName,
            tableName: tbName,
        }
    });

    public columnsExpr = this.RULE(RULES.columnsExpr, () => {
        const columns = this.OR([
            { ALT: () => this.SUBRULE(this.newColumns) },
            { ALT: () => this.SUBRULE(this.tableFieldExpr) },
            { ALT: () => this.SUBRULE(this.tableExpr) },
        ]);
        if (!Array.isArray(columns)) {
            return [columns];
        }
        return columns;
    });

    public tableExpr = this.RULE(RULES.tableExpr, () => {
        const talbeName = this.SUBRULE(this.identifier);
        return {
            type: 'TABLE',
            aliasName: talbeName,
        };
    });

    public identifier = this.RULE(RULES.Identifier, () => {
        const token = this.CONSUME(Identifier);
        return token.image;
    });

    public tableFieldExpr = this.RULE(RULES.tableFieldExpr, () => {
        const table = this.SUBRULE(this.tableExpr);
        this.CONSUME(DOT);
        const field = this.SUBRULE(this.identifier);
        return {
            type: 'TABLE_FIELD',
            aliasTableName: table.aliasName,
            field: field,
            aliasFieldName: field,
        };
    });

    public newColumns = this.RULE(RULES.newColumns, () => {
        const columns: any[] = [];
        this.CONSUME(NEW);
        this.CONSUME(LBRACE);
        this.MANY_SEP({
            SEP: COMMA,
            DEF: () => {
                const column = this.SUBRULE(this.fetchColumn);
                columns.push(column);
            },
        });
        this.CONSUME(RBRACE);
        return columns;
    });

    public fetchColumn = this.RULE(RULES.fetchColumn, () => {
        return this.OR([
            { ALT: () => this.SUBRULE(this.columnEqualExpr) },
            { ALT: () => this.SUBRULE(this.tableFieldExpr) },
        ]);
    });

    public columnEqualExpr = this.RULE(RULES.columnEqualExpr, () => {
        const aliasField = this.SUBRULE(this.identifier);
        this.CONSUME(ASSIGN);
        const tableField = this.SUBRULE(this.tableFieldExpr);

        return {
            ...tableField,
            aliasFieldName: aliasField,
        };
    });

    public integer = this.RULE(RULES.integer, () => {
        return this.CONSUME(Integer).image;
    });


    public integerExpr = this.RULE(RULES.integerExpr, () => {
        const integer = this.CONSUME(Integer).image;
        return {
            type: 'INTEGER',
            value: +integer,
        }
    });

    // === 優先順序 BEGIN ===
    public extractExpressions = this.RULE(RULES.extractExpressions, () => {
        return this.SUBRULE(this.extractOrExpr);
    });

    public extractOrExpr = this.RULE(RULES.extractOrExpr, () => {
        let left = this.SUBRULE(this.extractAndExpr);
        this.MANY(() => {
            const op = this.CONSUME(OR).image;
            const right = this.SUBRULE2(this.extractAndExpr);
            left = {
                type: 'OPERATOR',
                left: left,
                op: op,
                right: right,
            };
        });
        return left;
    });

    public extractAndExpr = this.RULE(RULES.extractAndExpr, () => {
        let left = this.SUBRULE(this.extractCompareExpr);
        this.MANY(() => {
            const op = this.CONSUME(AND).image;
            const right = this.SUBRULE2(this.extractCompareExpr);
            left = {
                type: 'OPERATOR',
                left: left,
                op: op,
                right: right,
            };
        });
        return left;
    });

    public extractCompareExpr = this.RULE(RULES.extractCompareExpr, () => {
        let left = this.SUBRULE(this.extractAtomExpr);
        this.MANY(() => {
            const op = this.SUBRULE(this.compareOper);
            const right = this.SUBRULE2(this.extractAtomExpr);
            left = {
                type: 'CONDITION',
                left: left,
                op: op,
                right: right,
            };
        });
        return left;
    });

    public compareOper = this.RULE(RULES.compareOper, () => {
        return this.OR([
            { ALT: () => this.CONSUME(GREATER_EQUAL) },
            { ALT: () => this.CONSUME(LESS_EQUAL) },
            { ALT: () => this.CONSUME(EQUAL) },
            { ALT: () => this.CONSUME(NOT_EQUAL) },
            { ALT: () => this.CONSUME(GREATER_THAN) },
            { ALT: () => this.CONSUME(LESS_THAN) },
        ]).image;
    });

    public extractAtomExpr = this.RULE(RULES.extractAtomExpr, () => {
        return this.OR([
            { ALT: () => this.SUBRULE(this.parenthesisExpr) },
            { ALT: () => this.SUBRULE(this.tableFieldExpr) },
            { ALT: () => this.CONSUME(Float) },
            { ALT: () => this.SUBRULE(this.integerExpr) },
        ])
    });

    public parenthesisExpr = this.RULE(RULES.parenthesisExpr, () => {
        const $: any = this;
        this.CONSUME(LPAREN);
        const expValue = this.SUBRULE($.extractExpressions);
        this.CONSUME(RPAREN);
        return {
            type: 'PARENTHESIS',
            expr: expValue,
        };
    });
    //=== 優先順序 END ===

    public whereExpr = this.RULE(RULES.whereExpr, () => {
        this.CONSUME(WHERE);
        return this.SUBRULE(this.extractExpressions);
    });
}

export function parseLinq(text: string) {
    const lexResult = linqLexer.tokenize(text);
    const parser = new LinqParserEmbedded();
    parser.input = lexResult.tokens;
    const value = parser.selectExpression();
    return {
        value: value,
        lexResult: lexResult,
        parseErrors: parser.errors,
    };
}
