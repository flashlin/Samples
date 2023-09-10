/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-unused-vars */
//https://github.com/kevinkhill/ts-tokenizr
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
    try {
      this._lexer.input(text);
      const tokens = this._lexer.tokens();
      tokens.pop();
      return tokens.map((x) => {
        if (x.text == "\n") {
          x.value = " ";
        } else if (x.text == "\r") {
          x.value = " ";
        }
        return x;
      });
    } catch (e) {
      console.error(
        `${e.message} pos=${e.pos}, '${text.substring(e.pos, e.pos + 10)}'`
      );
      throw e;
    }
  }

  private init() {
    this._lexer.rule(/[ \t\r\n]+/, (ctx, match) => {
      ctx.accept("spaces");
    });

    this._lexer.rule(/\/\/[^\r\n]*\r?\n/, (ctx, match) => {
      ctx.ignore();
    });

    this._lexer.rule(/"((\\")|[^"])*"/, (ctx, match) => {
      ctx.accept("string");
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
      symbolsRegExp([
        "!",
        "~",
        "@",
        "^",
        "+",
        "-",
        "*",
        "/",
        "_",
        "`",
        "=",
        ",",
        ".",
        "(",
        ")",
        "[",
        "]",
        "<",
        ">",
        "{",
        "}",
        '"',
      ]),
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
