import { TSQLLexer } from '@/generated/TSQLLexer';
import type { TSQLListener } from '@/generated/TSQLListener';
import { SelectColumnContext, SelectColumnListContext, SelectStatementContext, TSQLParser } from '@/generated/TSQLParser';
import { CommonTokenStream, CharStreams } from 'antlr4ts';
import { ParseTreeWalker } from 'antlr4ts/tree/ParseTreeWalker';

import type { TSQLVisitor } from '@/generated/TSQLVisitor';
import { AbstractParseTreeVisitor } from 'antlr4ts/tree/AbstractParseTreeVisitor';

export class TsqlExprListener implements TSQLListener { }

interface TsqlExpr { }

class TsqlExprVisitor extends AbstractParseTreeVisitor<TsqlExpr> implements TSQLVisitor<TsqlExpr> {
    defaultResult() {
        return 0;
    }

    visitSelectStatement(ctx: SelectStatementContext) {
        return {
            select: this.visit(ctx.selectColumnList()),
            from: this.visit(ctx.fromClause()),
            //where: this.visit(ctx.whereClause()),
            //groupBy: this.visit(ctx.groupByClause()),
            //having: this.visit(ctx.havingClause()),
        }
    }

    visitSelectColumnList(ctx: SelectColumnListContext) {
        const columns = ctx.selectColumn().map(column => {
            return {
                name: column.ID()[0].text
            };
        })
        return columns;
    }

    visitSelectColumn(ctx: SelectColumnContext) {
        return ctx.ID()[0].text;
    }
}

export class Tsql {
    parse1(sql: string) {
        const charStream = CharStreams.fromString(sql);
        const lexer = new TSQLLexer(charStream);
        const tokenStream = new CommonTokenStream(lexer);
        const parser = new TSQLParser(tokenStream);

        const startContext = parser.start();
        ParseTreeWalker.DEFAULT.walk(new TsqlExprListener(), startContext);
    }

    parse(sql: string) {
        const tree = this.compilation(sql);
        const tsqlVisitor = new TsqlExprVisitor();
        const expr = tsqlVisitor.visit(tree);
        console.log(expr);
    }

    compilation(sql: string) {
        const inputStream = CharStreams.fromString(sql);
        const lexer = new TSQLLexer(inputStream);
        const tokenStream = new CommonTokenStream(lexer);
        const parser = new TSQLParser(tokenStream);
        const tree = parser.start();
        return tree;
    }
}
