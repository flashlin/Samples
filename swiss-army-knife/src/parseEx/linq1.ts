//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser, EmbeddedActionsParser } from "chevrotain";

const RULES = {
    IDENTIFIER: "IDENTIFIER",
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
    tableFieldColumn: "tableFieldColumn",
    columnList: "columnList",
    newColumns: "newColumns",
    columnEqual: "columnEqual",
    compareOper: "compareOper",
    extractParentExpression: "extractParentExpression",
    extractExpressions: "extractExpressions",
    extractCompareExprs: "extractCompareExprs",
    extractCompareExpr: "extractCompareExpr",
    extractAndOrExprs: "extractAndOrExprs",
    extractAndExpr: "extractAndExpr",
    extractOrExpr: "extractOrExpr",
    expr: "expr",
    fetchColumn: "fetchColumn",
    whereExpr: "whereExpr",
    identifier: "identifier",
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

class LinqParser extends CstParser {

    constructor() {
        super(allTokens, { nodeLocationTracking: "onlyOffset" })  //"onlyOffset"
        this.performSelfAnalysis()
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        let where;
        this.CONSUME(FROM);
        const aliasName = this.SUBRULE(this.aliaName);
        this.CONSUME(IN);
        const source = this.SUBRULE(this.sourceClause);
        this.OPTION(() => {
            where = this.SUBRULE(this.whereExpr);
        });
        this.CONSUME(SELECT);
        const columns = this.SUBRULE(this.columnsExpr);

        return {
            type: "SELECT_CLAUSE",
            source: source,
            aliasName: aliasName,
            columns: columns,
            where: where
        };

    });

    public aliaName = this.RULE(RULES.aliaName, () => {
        return this.CONSUME(Identifier);
    });

    public sourceClause = this.RULE(RULES.sourceClause, () => {
        return this.OR([
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
        return this.OR([
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

    public extractParentExpression = this.RULE(RULES.extractParentExpression, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.tableFieldColumn) },
            { ALT: () => this.CONSUME(Float) },
            { ALT: () => this.CONSUME(Integer) },
        ])
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

    public extractExpressions = this.RULE(RULES.extractExpressions, () => {
        this.SUBRULE(this.extractParentExpression);
        this.OPTION(() => this.OR([
            { ALT: () => this.SUBRULE(this.extractCompareExpr) },
            { ALT: () => this.SUBRULE(this.extractAndExpr) },
            { ALT: () => this.SUBRULE(this.extractOrExpr) }
        ]));
    });

    public extractCompareExpr = this.RULE(RULES.extractCompareExpr, () => {
        this.SUBRULE(this.compareOper);
        this.SUBRULE(this.extractExpressions);
    });

    public extractAndExpr = this.RULE(RULES.extractAndExpr, () => {
        this.CONSUME(AND);
        this.SUBRULE(this.extractExpressions);
    });

    public extractOrExpr = this.RULE(RULES.extractOrExpr, () => {
        this.CONSUME(OR);
        this.SUBRULE(this.extractExpressions);
    });

    public whereExpr = this.RULE(RULES.whereExpr, () => {
        this.CONSUME(WHERE);
        return this.SUBRULE(this.extractExpressions);
    });
}


class LinqParserEmbedded extends EmbeddedActionsParser {
    constructor() {
        super(allTokens);
        this.performSelfAnalysis();
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        let where;
        this.CONSUME(FROM);
        const aliasName = this.SUBRULE(this.aliaName);
        this.CONSUME(IN);
        const source = this.SUBRULE(this.sourceClause);
        this.OPTION(() => {
            where = this.SUBRULE(this.whereExpr);
        });
        this.CONSUME(SELECT);
        const columns = this.SUBRULE(this.columnsExpr);

        return {
            type: "SELECT_CLAUSE",
            source: source,
            aliasName: aliasName,
            columns: columns,
            where: where
        };

    });

    public aliaName = this.RULE(RULES.aliaName, () => {
        return this.CONSUME(Identifier);
    });

    public sourceClause = this.RULE(RULES.sourceClause, () => {
        return this.OR([
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
        return this.OR([
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

    public extractParentExpression = this.RULE(RULES.extractParentExpression, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.tableFieldColumn) },
            { ALT: () => this.CONSUME(Float) },
            { ALT: () => this.CONSUME(Integer) },
        ])
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

    public extractExpressions = this.RULE(RULES.extractExpressions, () => {
        this.SUBRULE(this.extractParentExpression);
        this.OPTION(() => this.OR([
            { ALT: () => this.SUBRULE(this.extractCompareExpr) },
            { ALT: () => this.SUBRULE(this.extractAndExpr) },
            { ALT: () => this.SUBRULE(this.extractOrExpr) }
        ]));
    });

    public extractCompareExpr = this.RULE(RULES.extractCompareExpr, () => {
        this.SUBRULE(this.compareOper);
        this.SUBRULE(this.extractExpressions);
    });

    public extractAndExpr = this.RULE(RULES.extractAndExpr, () => {
        this.CONSUME(AND);
        this.SUBRULE(this.extractExpressions);
    });

    public extractOrExpr = this.RULE(RULES.extractOrExpr, () => {
        this.CONSUME(OR);
        this.SUBRULE(this.extractExpressions);
    });

    public whereExpr = this.RULE(RULES.whereExpr, () => {
        this.CONSUME(WHERE);
        return this.SUBRULE(this.extractExpressions);
    });
}

export function parseLinqEmbedded(text: string) {
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
        const where = this.visit(ctx.whereExpr);
        return {
            type: "SELECT_CLAUSE",
            source: source,
            aliaName: aliaName,
            columns: columns,
            where: where
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

    whereExpr(ctx: any) {
        const list = this.visit(ctx.extractExpressions);
        console.log('where2', list);
        return {
            type: 'WHERE_CLAUSE',
        };
    }

    extractExpressions(ctx: any) {
        console.log('extractExpression=', ctx)
        return {

        };
    }

    extractCompareExpr(ctx: any) {
        console.log('extractCoompare=', ctx);
        return {

        };
    }
}


const linqVisitorWithDefaults = new LinqExprVisitorWithDefaults();
export function parseLinq(text: string) {
    const lexResult = linqLexer.tokenize(text);
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