//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser } from "chevrotain";

const RULES = {
    IDENTIFIER: "IDENTIFIER",
    selectExpression: "selectExpression",
    columnList: "columnList",
    identifier: "identifier",
};

const StringDoubleQuote = createToken({ name: "StringDoubleQuote", pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: "StringSimpleQuote", pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const SELECT = createToken({ name: "Select", pattern: /(SELECT|select)/ });
const And = createToken({ name: "And", pattern: /(AND|and)/ });
const Or = createToken({ name: "Or", pattern: /(OR|or)/ });
const Not = createToken({ name: "Not", pattern: /(NOT|not)/ });
const Colon = createToken({ name: "Colon", pattern: /:/ });

const WhiteSpace = createToken({
    name: "WhiteSpace",
    pattern: /[ \t\n\r]+/,
    group: Lexer.SKIPPED
});

const allTokens = [
    WhiteSpace,
    Colon,
    And,
    Or,
    Not,
    SELECT,
    Identifier,
    StringDoubleQuote,
    StringSimpleQuote
];

const TSqlLexer = new Lexer(allTokens);

class TSqlParser extends CstParser {
    private static INSTANCE: TSqlParser | undefined;
    public static get(): TSqlParser {
        if (TSqlParser.INSTANCE === undefined) {
            TSqlParser.INSTANCE = new TSqlParser();
        }
        return TSqlParser.INSTANCE;
    }

    constructor() {
        super(allTokens, { nodeLocationTracking: "onlyOffset" })
        this.performSelfAnalysis()
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        this.CONSUME(SELECT);
        this.SUBRULE(this.columnList);
    });

    public columnList = this.RULE(RULES.columnList, () => {
        this.MANY(() => this.OR([
            { ALT: () => this.SUBRULE(this.identifier) },
        ]));
    });

    public identifier = this.RULE(RULES.identifier, () => {
        this.CONSUME(Identifier);
    });
}

const parserInstance = new TSqlParser();
const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();
const BaseTSqlVisitorWithDefaults = parserInstance.getBaseCstVisitorConstructorWithDefaults();

class TSqlExprVisitor extends BaseCstVisitor {
    constructor() {
        super();
        this.validateVisitor();
    }

    selectExpression(ctx: any) {
        console.log("select", ctx)
        const columns = ctx.Identifier.map((identToken: any) => identToken.image);
        return {
            type: "SELECT_CLAUSE",
            columns: columns,
        };
    }

    /* all Visit methods must go here */
}

class TSqlExprVisitorWithDefaults extends BaseTSqlVisitorWithDefaults {
    constructor() {
        super();
        this.validateVisitor();
    }
    /* Visit methods go here */

    selectExpression(ctx: any) {
        console.log("select2 ctx", ctx)
        console.log("select2", ctx.columnList)
        const columns = this.visit(ctx.columnList).columns;
        return {
            type: "SELECT_CLAUSE",
            columns: columns,
        };
    }

    columnList(ctx: any) {
        const columns = ctx.identifier.map((identToken: any) => this.visit(identToken));
        return {
            type: "COLUMN_LIST",
            columns: columns,
        };
    }

    identifier(ctx: any) {
        return {
            type: "IDENTIFIER",
            value: ctx.IDENTIFIER[0].image,
        }
    }
}

//const tsqlVisitor = new TSqlExprVisitor();
const tsqlVisitorWithDefaults = new TSqlExprVisitorWithDefaults();

export function parseTsql(text: string) {
    const lexResult = TSqlLexer.tokenize(text);
    parserInstance.input = lexResult.tokens;
    const cst = parserInstance.selectExpression();
    //const value = tsqlVisitor.visit(cst);
    const value = tsqlVisitorWithDefaults.visit(cst);
    console.log('v', value)
    return {
        value: value,
        lexResult: lexResult,
        parseErrors: parserInstance.errors,
    };
}