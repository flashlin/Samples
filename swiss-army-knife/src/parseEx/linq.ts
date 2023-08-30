//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser } from "chevrotain";

const RULES = {
    IDENTIFIER: "IDENTIFIER",
    selectExpression: "selectExpression",
    aliaName: "aliaName",
    sourceClause: "sourceClause",
    tableClause: "tableClause",
    databaseTableClause: "databaseTableClause",
    columnsExpr: "columnsExpr",
    tableExpr: "tableExpr",
    tableFieldColumn: "tableFieldColumn",
    columnList: "columnList",
    newColumns: "newColumns",
    columnEqual: "columnEqual",
    compareOper: "compareOper",
    compareExpr: "compareExpr",
    expr: "expr",
    fetchColumn: "fetchColumn",
    identifier: "identifier",
};

const StringDoubleQuote = createToken({ name: "StringDoubleQuote", pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: "StringSimpleQuote", pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const SELECT = createToken({ name: "select", pattern: /select/ });
const FROM = createToken({ name: "from", pattern: /from/ });
const WHERE = createToken({ name: "where", pattern: /where/ });
const IN = createToken({ name: "in", pattern: /in/ });
const AND = createToken({ name: "and", pattern: /(&&)/ });
const OR = createToken({ name: "or", pattern: /(\|\|)/ });
const NOT = createToken({ name: "not", pattern: /(\!)/ });
const NEW = createToken({ name: "new", pattern: /new/ });
const DOT = createToken({ name: "dot", pattern: /(\.)/ });
const ASSIGN = createToken({ name: "assign", pattern: /(\=)/ });
const LBRACE = createToken({ name: "left brace", pattern: /\{/ });
const RBRACE = createToken({ name: "right brace", pattern: /\}/ });
const COMMA = createToken({ name: "comma", pattern: /\,/ });
const GREATER_THAN = createToken({ name: ">", pattern: /\>/ });
const LESS_THAN = createToken({ name: "<", pattern: /\</ });
const EQUAL = createToken({ name: "==", pattern: /\==/ });
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

const LinqLexer = new Lexer(allTokens);

class LinqParser extends CstParser {

    constructor() {
        super(allTokens, { nodeLocationTracking: "onlyOffset" })
        this.performSelfAnalysis()
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        this.CONSUME(FROM);
        this.SUBRULE(this.aliaName);
        this.CONSUME(IN);
        this.SUBRULE(this.sourceClause);
        this.CONSUME(SELECT);
        this.SUBRULE(this.columnsExpr);
    });

    public aliaName = this.RULE(RULES.aliaName, () => {
        this.CONSUME(Identifier);
    });

    public sourceClause = this.RULE(RULES.sourceClause, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.databaseTableClause) },
            { ALT: () => this.SUBRULE(this.tableClause) },
        ]);
    });

    public tableClause = this.RULE(RULES.tableClause, () => {
        this.CONSUME(Identifier);
    });

    public databaseTableClause = this.RULE(RULES.databaseTableClause, () => {
        this.CONSUME(Identifier);
        this.CONSUME2(DOT);
        this.CONSUME3(Identifier);
    });

    public columnsExpr = this.RULE(RULES.columnsExpr, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.newColumns) },
            { ALT: () => this.SUBRULE(this.tableFieldColumn) },
            { ALT: () => this.SUBRULE(this.tableExpr) },
        ]);
    });

    public tableExpr = this.RULE(RULES.tableExpr, () => {
        this.CONSUME(Identifier);
    });

    public tableFieldColumn = this.RULE(RULES.tableFieldColumn, () => {
        this.SUBRULE(this.tableExpr);
        this.CONSUME(DOT);
        this.CONSUME(Identifier);
    });

    public newColumns = this.RULE(RULES.newColumns, () => {
        this.CONSUME(NEW);
        this.CONSUME(LBRACE);
        this.MANY_SEP({
            SEP: COMMA,
            DEF: () => this.SUBRULE(this.fetchColumn),
        });
        this.CONSUME(RBRACE);
    });

    public fetchColumn = this.RULE(RULES.fetchColumn, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.columnEqual) },
            { ALT: () => this.SUBRULE(this.tableFieldColumn) },
        ])
    });

    public columnEqual = this.RULE(RULES.columnEqual, () => {
        this.CONSUME(Identifier);
        this.CONSUME(ASSIGN);
        this.SUBRULE(this.tableFieldColumn);
    });

    public expr = this.RULE(RULES.expr, () => {
        this.SUBRULE(this.tableFieldColumn);
    });

    public compareOper = this.RULE(RULES.compareOper, () => {
        this.OR([
            { ALT: () => this.CONSUME(GREATER_EQUAL) },
            { ALT: () => this.CONSUME(LESS_EQUAL) },
            { ALT: () => this.CONSUME(EQUAL) },
            { ALT: () => this.CONSUME(NOT_EQUAL) },
            { ALT: () => this.CONSUME(GREATER_THAN) },
            { ALT: () => this.CONSUME(LESS_THAN) },
        ])
    });

    public compareExpr = this.RULE(RULES.compareExpr, () => {
        this.SUBRULE(this.expr);
        this.SUBRULE(this.compareOper);
        this.SUBRULE2(this.expr);
    });
}

const parser = new LinqParser();

//const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();
// class LinqExprVisitor extends BaseCstVisitor {
//     constructor() {
//         super();
//         this.validateVisitor();
//     }
//     /* all Visit methods must go here */
// }
//const _linqVisitor = new LinqExprVisitor();

export interface ITableFieldExpr {
    type: string;
    aliasTable: string;
    field: string;
    aliasField: string;
}


const BaseLinqVisitorWithDefaults = parser.getBaseCstVisitorConstructorWithDefaults();
class LinqExprVisitorWithDefaults extends BaseLinqVisitorWithDefaults {
    constructor() {
        super();
        this.validateVisitor();
    }

    selectExpression(ctx: any) {
        const aliaName = this.visit(ctx.aliaName).alias;
        const source = this.visit(ctx.sourceClause);
        const columns = this.visit(ctx.columnsExpr);
        return {
            type: "SELECT_CLAUSE",
            source: source,
            aliaName: aliaName,
            columns: columns
        };
    }

    aliaName(ctx: any) {
        return {
            type: 'ALIAS_NAME',
            alias: ctx.IDENTIFIER[0].image
        };
    }

    sourceClause(ctx: any) {
        if (ctx.tableClause) {
            return this.visit(ctx.tableClause);
        }
        return this.visit(ctx.databaseTableClause);
    }

    tableClause(ctx: any) {
        return {
            type: 'TABLE_CLAUSE',
            name: ctx.IDENTIFIER[0].image
        };
    }

    tableFieldColumn(ctx: any): ITableFieldExpr {
        const tableName = this.visit(ctx.tableExpr).name;
        const fieldName = ctx.IDENTIFIER[0].image;
        return {
            type: 'TABLE_FIELD',
            aliasTable: tableName,
            field: fieldName,
            aliasField: fieldName,
        };
    }

    tableExpr(ctx: any) {
        return {
            type: 'TABLE',
            name: ctx.IDENTIFIER[0].image,
        };
    }

    columnsExpr(ctx: any) {
        if (ctx.newColumns) {
            return this.visit(ctx.newColumns);
        }
        if (ctx.tableFieldColumn) {
            return [this.visit(ctx.tableFieldColumn)];
        }
        return [this.visit(ctx.tableExpr)];
    }

    newColumns(ctx: any) {
        const columns = ctx.fetchColumn.map((x: any) => this.visit(x));
        return columns;
    }

    fetchColumn(ctx: any) {
        if (ctx.tableFieldColumn) {
            return this.visit(ctx.tableFieldColumn);
        }
        return this.visit(ctx.columnEqual);
    }

    columnEqual(ctx: any): ITableFieldExpr {
        const aliasField = ctx.IDENTIFIER[0].image;
        const tableField = this.visit(ctx.tableFieldColumn);
        return {
            ...tableField,
            aliasField: aliasField,
        };
    }
}


const linqVisitorWithDefaults = new LinqExprVisitorWithDefaults();
export function parseLinq(text: string) {
    const lexResult = LinqLexer.tokenize(text);
    parser.input = lexResult.tokens;
    const cst = parser.selectExpression();
    //const value = tsqlVisitor.visit(cst);
    const value = linqVisitorWithDefaults.visit(cst);
    return {
        value: value,
        lexResult: lexResult,
        parseErrors: parser.errors,
    };
}