// Generated from ./src/antlr/TSQL.g4 by ANTLR 4.9.0-SNAPSHOT


import { ParseTreeVisitor } from "antlr4ts/tree/ParseTreeVisitor";

import { StartContext } from "./TSQLParser";
import { SelectStatementContext } from "./TSQLParser";
import { SelectColumnListContext } from "./TSQLParser";
import { SelectColumnContext } from "./TSQLParser";
import { FromClauseContext } from "./TSQLParser";
import { TableReferenceContext } from "./TSQLParser";


/**
 * This interface defines a complete generic visitor for a parse tree produced
 * by `TSQLParser`.
 *
 * @param <Result> The return type of the visit operation. Use `void` for
 * operations with no return type.
 */
export interface TSQLVisitor<Result> extends ParseTreeVisitor<Result> {
	/**
	 * Visit a parse tree produced by `TSQLParser.start`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitStart?: (ctx: StartContext) => Result;

	/**
	 * Visit a parse tree produced by `TSQLParser.selectStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitSelectStatement?: (ctx: SelectStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `TSQLParser.selectColumnList`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitSelectColumnList?: (ctx: SelectColumnListContext) => Result;

	/**
	 * Visit a parse tree produced by `TSQLParser.selectColumn`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitSelectColumn?: (ctx: SelectColumnContext) => Result;

	/**
	 * Visit a parse tree produced by `TSQLParser.fromClause`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFromClause?: (ctx: FromClauseContext) => Result;

	/**
	 * Visit a parse tree produced by `TSQLParser.tableReference`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitTableReference?: (ctx: TableReferenceContext) => Result;
}

