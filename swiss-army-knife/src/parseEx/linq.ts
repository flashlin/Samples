//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser } from "chevrotain";

const RULES = {
    IDENTIFIER: "IDENTIFIER",
    selectExpression: "selectExpression",
    columnList: "columnList",
    identifier: "identifier",
    sourceClause: "sourceClause",
};

const StringDoubleQuote = createToken({ name: "StringDoubleQuote", pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: "StringSimpleQuote", pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const SELECT = createToken({ name: "Select", pattern: /(SELECT|select)/ });
const FROM = createToken({ name: "From", pattern: /(FROM|from)/ });
const AND = createToken({ name: "And", pattern: /(AND|and)/ });
const OR = createToken({ name: "Or", pattern: /(OR|or)/ });
const NOT = createToken({ name: "Not", pattern: /(NOT|not)/ });
const Colon = createToken({ name: "Colon", pattern: /:/ });

const WhiteSpace = createToken({
    name: "WhiteSpace",
    pattern: /[ \t\n\r]+/,
    group: Lexer.SKIPPED
});

const allTokens = [
    WhiteSpace,
    Colon,
    FROM,
    AND,
    OR,
    NOT,
    SELECT,
    Identifier,
    StringDoubleQuote,
    StringSimpleQuote
];

const LinqLexer = new Lexer(allTokens);

class LinqParser extends CstParser {
    private static INSTANCE: LinqParser | undefined;
    public static get(): LinqParser {
        if (LinqParser.INSTANCE === undefined) {
            LinqParser.INSTANCE = new LinqParser();
        }
        return LinqParser.INSTANCE;
    }

    constructor() {
        super(allTokens, { nodeLocationTracking: "onlyOffset" })
        this.performSelfAnalysis()
    }

    public selectExpression = this.RULE(RULES.selectExpression, () => {
        this.CONSUME(SELECT);
        this.SUBRULE(this.columnList);
        this.CONSUME(FROM);
        this.SUBRULE(this.sourceClause);
    });

    public columnList = this.RULE(RULES.columnList, () => {
        this.MANY(() => this.OR([
            { ALT: () => this.SUBRULE(this.identifier) },
        ]));
    });

    public sourceClause = this.RULE(RULES.sourceClause, () => {
        this.OR([
            { ALT: () => this.SUBRULE(this.identifier) },
        ]);
    });

    public identifier = this.RULE(RULES.identifier, () => {
        this.CONSUME(Identifier);
    });
}

const parserInstance = new LinqParser();
const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();
const BaseLinqVisitorWithDefaults = parserInstance.getBaseCstVisitorConstructorWithDefaults();

class LinqExprVisitor extends BaseCstVisitor {
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

class LinqExprVisitorWithDefaults extends BaseLinqVisitorWithDefaults {
    constructor() {
        super();
        this.validateVisitor();
    }
    /* Visit methods go here */

    selectExpression(ctx: any) {
        const columns = this.visit(ctx.columnList).columns;
        const sourceClause = this.visit(ctx.sourceClause);
        return {
            type: "SELECT_CLAUSE",
            columns: columns,
            sourceClause: sourceClause.value,
        };
    }

    columnList(ctx: any) {
        const columns = ctx.identifier.map((identToken: any) => this.visit(identToken));
        return {
            type: "COLUMN_LIST",
            columns: columns,
        };
    }

    sourceClause(ctx: any) {
        return {
            type: "SOURCE_CLAUSE",
            value: ctx.identifier.map((identToken: any) => this.visit(identToken)),
        }
    }

    identifier(ctx: any) {
        return {
            type: "IDENTIFIER",
            value: ctx.IDENTIFIER[0].image,
        }
    }
}

const _linqVisitor = new LinqExprVisitor();
const linqVisitorWithDefaults = new LinqExprVisitorWithDefaults();

export function parseLinq(text: string) {
    const lexResult = LinqLexer.tokenize(text);
    parserInstance.input = lexResult.tokens;
    const cst = parserInstance.selectExpression();
    //const value = tsqlVisitor.visit(cst);
    const value = linqVisitorWithDefaults.visit(cst);
    return {
        value: value,
        lexResult: lexResult,
        parseErrors: parserInstance.errors,
    };
}