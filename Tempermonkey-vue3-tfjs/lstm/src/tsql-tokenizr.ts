/* eslint-disable @typescript-eslint/no-unused-vars */
import { Tokenizr } from "ts-tokenizr";

export const keywords = [
  "ADD",
  "EXTERNAL",
  "PROCEDURE",
  "ALL",
  "FETCH",
  "PUBLIC",
  "ALTER",
  "FILE",
  "RAISERROR",
  "AND",
  "FILLFACTOR",
  "READ",
  "ANY",
  "FOR",
  "READTEXT",
  "AS",
  "FOREIGN",
  "RECONFIGURE",
  "ASC",
  "FREETEXT",
  "REFERENCES",
  "AUTHORIZATION",
  "FREETEXTTABLE",
  "REPLICATION",
  "BACKUP",
  "FROM",
  "RESTORE",
  "BEGIN",
  "FULL",
  "RESTRICT",
  "BETWEEN",
  "FUNCTION",
  "RETURN",
  "BREAK",
  "GOTO",
  "REVERT",
  "BROWSE",
  "GRANT",
  "REVOKE",
  "BULK",
  "GROUP",
  "RIGHT",
  "BY",
  "HAVING",
  "ROLLBACK",
  "CASCADE",
  "HOLDLOCK",
  "ROWCOUNT",
  "CASE",
  "IDENTITY",
  "ROWGUIDCOL",
  "CHECK",
  "IDENTITY_INSERT",
  "RULE",
  "CHECKPOINT",
  "IDENTITYCOL",
  "SAVE",
  "CLOSE",
  "IF",
  "SCHEMA",
  "CLUSTERED",
  "IN",
  "SECURITYAUDIT",
  "COALESCE",
  "INDEX",
  "SELECT",
  "COLLATE",
  "INNER",
  "SEMANTICKEYPHRASETABLE",
  "COLUMN",
  "INSERT",
  "SEMANTICSIMILARITYDETAILSTABLE",
  "COMMIT",
  "INTERSECT",
  "SEMANTICSIMILARITYTABLE",
  "COMPUTE",
  "INTO",
  "SESSION_USER",
  "CONSTRAINT",
  "IS",
  "SET",
  "CONTAINS",
  "JOIN",
  "SETUSER",
  "CONTAINSTABLE",
  "KEY",
  "SHUTDOWN",
  "CONTINUE",
  "KILL",
  "SOME",
  "CONVERT",
  "LEFT",
  "STATISTICS",
  "CREATE",
  "LIKE",
  "SYSTEM_USER",
  "CROSS",
  "LINENO",
  "TABLE",
  "CURRENT",
  "LOAD",
  "TABLESAMPLE",
  "CURRENT_DATE",
  "MERGE",
  "TEXTSIZE",
  "CURRENT_TIME",
  "NATIONAL",
  "THEN",
  "CURRENT_TIMESTAMP",
  "NOCHECK",
  "TO",
  "CURRENT_USER",
  "NONCLUSTERED",
  "TOP",
  "CURSOR",
  "NOT",
  "TRAN",
  "DATABASE",
  "NULL",
  "TRANSACTION",
  "DBCC",
  "NULLIF",
  "TRIGGER",
  "DEALLOCATE",
  "OF",
  "TRUNCATE",
  "DECLARE",
  "OFF",
  "TRY_CONVERT",
  "DEFAULT",
  "OFFSETS",
  "TSEQUAL",
  "DELETE",
  "ON",
  "UNION",
  "DENY",
  "OPEN",
  "UNIQUE",
  "DESC",
  "OPENDATASOURCE",
  "UNPIVOT",
  "DISK",
  "OPENQUERY",
  "UPDATE",
  "DISTINCT",
  "OPENROWSET",
  "UPDATETEXT",
  "DISTRIBUTED",
  "OPENXML",
  "USE",
  "DOUBLE",
  "OPTION",
  "USER",
  "DROP",
  "OR",
  "VALUES",
  "DUMP",
  "ORDER",
  "VARYING",
  "ELSE",
  "OUTER",
  "VIEW",
  "END",
  "OVER",
  "WAITFOR",
  "ERRLVL",
  "PERCENT",
  "WHEN",
  "ESCAPE",
  "PIVOT",
  "WHERE",
  "EXCEPT",
  "PLAN",
  "WHILE",
  "EXEC",
  "PRECISION",
  "WITH",
  "EXECUTE",
  "PRIMARY",
  "WITHIN GROUP",
  "EXISTS",
  "PRINT",
  "WRITETEXT",
  "EXIT",
  "PROC",
];

keywords.sort().reverse();
const keywordRegex = keywords.map((x) => `(${x})`).join("|");

export class TSqlTokenizr {
  private _lexer: Tokenizr = new Tokenizr({
    debug: false,
  });

  constructor() {
    this.init();
  }

  tokens(text: string) {
    this._lexer.input(text);
    try {
      const tokens = this._lexer.tokens();
      tokens.pop();
      return tokens.map((x) => {
        if (x.text == "\r") {
          x.value = " ";
        }
        return x;
      });
    } catch (e) {
      console.error(
        `${e.message} pos=${e.pos} '${text.substring(e.pos, e.pos + 10)}'`
      );
      throw e;
    }
  }

  private init() {
    this._lexer.rule(/[ \t\r\n]+/, (ctx, match) => {
      ctx.accept("spaces");
    });

    this._lexer.rule(/--[^\r\n]*\r?\n/, (ctx, match) => {
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

    this._lexer.rule(/'(\\'|[^'])*'/, (ctx, match) => {
      ctx.accept("string");
    });

    this._lexer.rule(/[.,()[\]]/, (ctx, match) => {
      ctx.accept("symbol");
    });

    this._lexer.rule(/[*+\-%]/, (ctx, match) => {
      ctx.accept("operator");
    });

    this._lexer.rule(/(<>)|(!=)|(<=)|(>=)|<|>/, (ctx, match) => {
      ctx.accept("compare");
    });
  }
}
