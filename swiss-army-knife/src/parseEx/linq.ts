//https://github.com/Chevrotain/chevrotain/blob/master/examples/grammars/calculator/calculator_pure_grammar.js
import { createToken, tokenMatcher, Lexer, CstParser } from "chevrotain";

const RULES = {
    IDENTIFIER: "IDENTIFIER",
    selectExpression: "selectExpression",
    aliaName: "aliasName",
    databaseClause: "databaseClause",
    selectExpr: "selectExpr",
    tableExpr: "tableExpr",
    tableFieldExpr: "tableFieldExpr",
    columnList: "columnList",
    identifier: "identifier",
    sourceClause: "sourceClause",
};

const StringDoubleQuote = createToken({ name: "StringDoubleQuote", pattern: /"[^"\\]*(?:\\.[^"\\]*)*"/ });
const StringSimpleQuote = createToken({ name: "StringSimpleQuote", pattern: /'[^'\\]*(?:\\.[^'\\]*)*'/ });
const Identifier = createToken({ name: RULES.IDENTIFIER, pattern: /[a-zA-Z_]\w*/ });
const SELECT = createToken({ name: "select", pattern: /select/ });
const FROM = createToken({ name: "from", pattern: /(from)/ });
const IN = createToken({ name: "in", pattern: /in/ });
const AND = createToken({ name: "and", pattern: /(&&)/ });
const OR = createToken({ name: "or", pattern: /(\|\|)/ });
const NOT = createToken({ name: "not", pattern: /(\!)/ });
const DOT = createToken({ name: "not", pattern: /(\.)/ });

const WhiteSpace = createToken({
    name: "WhiteSpace",
    pattern: /[ \t\n\r]+/,
    group: Lexer.SKIPPED
});

const allTokens = [
    WhiteSpace,
    SELECT,
    FROM,
    AND,
    NOT,
    IN,
    OR,
    DOT,
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
        this.CONSUME(FROM);
        this.CONSUME(Identifier);
        this.SUBRULE(this.aliaName);
    });

    public aliaName = this.RULE(RULES.aliaName, () => {
        this.CONSUME(Identifier);
    });

    public databaseClause = this.RULE(RULES.databaseClause, () => {
        this.CONSUME(Identifier);
        this.CONSUME2(DOT);
        this.CONSUME3(Identifier);
    });

    // public selectExpr = this.RULE(RULES.selectExpr, () => {
    //     this.OR([
    //         { ALT: () => this.SUBRULE(this.tableFieldExpr) },
    //         { ALT: () => this.SUBRULE(this.tableExpr) },
    //     ]);
    // });

    // public tableExpr = this.RULE(RULES.tableExpr, () => {
    //     this.CONSUME(Identifier);
    // });

    // public tableFieldExpr = this.RULE(RULES.tableFieldExpr, () => {
    //     this.SUBRULE(this.tableExpr);
    //     this.CONSUME(DOT);
    //     this.CONSUME(Identifier);
    // });

    // public identifier = this.RULE(RULES.identifier, () => {
    //     this.CONSUME(Identifier);
    // });
}

const parserInstance = new LinqParser();

//const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();
// class LinqExprVisitor extends BaseCstVisitor {
//     constructor() {
//         super();
//         this.validateVisitor();
//     }
//     /* all Visit methods must go here */
// }
//const _linqVisitor = new LinqExprVisitor();


const BaseLinqVisitorWithDefaults = parserInstance.getBaseCstVisitorConstructorWithDefaults();
class LinqExprVisitorWithDefaults extends BaseLinqVisitorWithDefaults {
    constructor() {
        super();
        this.validateVisitor();
    }

    selectExpression(ctx: any) {
        console.log('a1')
        const aliaName = this.visit(ctx);
        console.log('a', aliaName)
        return {
            type: "SELECT_CLAUSE",
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