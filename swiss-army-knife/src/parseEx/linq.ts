//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser, EmbeddedActionsParser } from "chevrotain";

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
    columnList: "columnList",
    newColumns: "newColumns",
    columnEqualExpr: "columnEqual",
    compareOper: "compareOper",
    extractAtomExpr: "extractParentExpression",
    extractExpressions: "extractExpressions",
    extractCompareExprs: "extractCompareExprs",
    extractCompareExpr: "extractCompareExpr",
    extractAndOrExprs: "extractAndOrExprs",
    extractAndExpr: "extractAndExpr",
    extractOrExpr: "extractOrExpr",
    integer: "integer",
    expr: "expr",
    fetchColumn: "fetchColumn",
    whereExpr: "whereExpr",
    Identifier: "identifier",
    IDENTIFIER: "Identifier",
    FLOAT: "Float",
    INTEGER: "Integer",
};

const StringDoubleQuote = createToken({ name: "StringDoubleQuote", pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: "StringSimpleQuote", pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const Float = createToken({ name: RULES.FLOAT, pattern: /\d+\.\d*/ });
const Integer = createToken({ name: RULES.INTEGER, pattern: /\d+/ });
const SELECT = createToken({ name: "select", pattern: /select/ });
const FROM = createToken({ name: "from", pattern: /from/ });
const WHERE = createToken({ name: "where", pattern: /where/ });
const IN = createToken({ name: "in", pattern: /in/ });
const AND = createToken({ name: RULES.AND, pattern: /(&&)/ });
const OR = createToken({ name: "or", pattern: /(\|\|)/ });
const NOT = createToken({ name: "not", pattern: /(\!)/ });
const NEW = createToken({ name: "new", pattern: /new/ });
const DOT = createToken({ name: "dot", pattern: /(\.)/ });
const ASSIGN = createToken({ name: "assign", pattern: /(\=)/ });
const LBRACE = createToken({ name: "left brace", pattern: /\{/ });
const RBRACE = createToken({ name: "right brace", pattern: /\}/ });
const COMMA = createToken({ name: "comma", pattern: /\,/ });
const GREATER_THAN = createToken({ name: RULES.GREATER_THAN, pattern: /\>/ });
const LESS_THAN = createToken({ name: RULES.LESS_THAN, pattern: /\</ });
const EQUAL = createToken({ name: "==", pattern: /\=\=/ });
const GREATER_EQUAL = createToken({ name: ">=", pattern: /\>\=/ });
const LESS_EQUAL = createToken({ name: "<=", pattern: /\<\=/ });
const NOT_EQUAL = createToken({ name: "!=", pattern: /\!\=/ });

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
    COMMA,
    ASSIGN,
    GREATER_THAN,
    LESS_THAN,
    Identifier,
    StringDoubleQuote,
    StringSimpleQuote
];

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
        }
    });

    public databaseTableClause = this.RULE(RULES.databaseTableClause, () => {
        this.CONSUME(Identifier);
        this.CONSUME2(DOT);
        this.CONSUME3(Identifier);
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

    public compareOper = this.RULE(RULES.compareOper, () => {
        return this.OR([
            { ALT: () => this.CONSUME(GREATER_EQUAL).image },
            { ALT: () => this.CONSUME(LESS_EQUAL).image },
            { ALT: () => this.CONSUME(EQUAL).image },
            { ALT: () => this.CONSUME(NOT_EQUAL).image },
            { ALT: () => this.CONSUME(GREATER_THAN).image },
            { ALT: () => this.CONSUME(LESS_THAN).image },
        ])
    });

    public extractExpressions = this.RULE(RULES.extractExpressions, () => {
        return this.SUBRULE(this.extractOrExpr);
    });

    public extractOrExpr = this.RULE(RULES.extractOrExpr, () => {
        let left = this.SUBRULE(this.extractAndExpr);
        this.MANY(() => {
            const op = this.CONSUME(OR);
            const right = this.SUBRULE2(this.extractAndExpr);
            left = {
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
            const op = this.CONSUME(AND);
            const right = this.SUBRULE2(this.extractCompareExpr);
            left = {
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

    public extractAtomExpr = this.RULE(RULES.extractAtomExpr, () => {
        return this.OR([
            { ALT: () => this.SUBRULE(this.tableFieldExpr) },
            { ALT: () => this.CONSUME(Float) },
            { ALT: () => this.SUBRULE(this.integer) },
        ])
    });


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
