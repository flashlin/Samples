/* eslint-disable @typescript-eslint/no-unused-vars */
import { keywordsRegExp, symbolsRegExp } from "@/tokenizr-utils";
import { Tokenizr } from "ts-tokenizr";

export const keywords = [
  "select",
  "from",
  "into",
  "equals",
  "group",
  "by",
  "join",
  "in",
];

const keywordRegex = keywordsRegExp(keywords);

export class LinqTokenizr {
  private _lexer: Tokenizr = new Tokenizr({
    debug: false,
  });

  constructor() {
    this.init();
  }

  tokens(text: string) {
    this._lexer.input(text);
    const tokens = this._lexer.tokens();
    tokens.pop();
    return tokens;
  }

  private init() {
    this._lexer.rule(/[ \t\r\n]+/, (ctx, match) => {
      ctx.ignore();
    });

    this._lexer.rule(/\/\/[^\r\n]*\r?\n/, (ctx, match) => {
      ctx.ignore();
    });

    this._lexer.rule(new RegExp(keywordRegex, "i"), (ctx, match) => {
      ctx.accept("keyword");
    });

    this._lexer.rule(/[a-zA-Z_][a-zA-Z0-9_]*/, (ctx, match) => {
      ctx.accept("identifier");
    });

    this._lexer.rule(/[0-9]+(\.[0-9]+)?/, (ctx, match) => {
      ctx.accept("number");
    });

    this._lexer.rule(
      symbolsRegExp([".", "(", ")", "[", "]", "<", ">"]),
      (ctx, match) => {
        ctx.accept("symbol");
      }
    );

    const operators = [
      "==",
      "!=",
      "<",
      ">",
      ">=",
      "<=",
      "~",
      "!",
      "=",
      "&&",
      "&",
      "|",
      "+",
      "-",
      "*",
      "/",
    ];

    this._lexer.rule(symbolsRegExp(operators), (ctx, match) => {
      ctx.accept("operator");
    });
  }
}
