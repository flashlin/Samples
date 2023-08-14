// Generated from ./src/antlr/TSQL.g4 by ANTLR 4.9.0-SNAPSHOT


import { ParseTreeListener } from "antlr4ts/tree/ParseTreeListener";

import { StartContext } from "./TSQLParser";
import { SelectStatementContext } from "./TSQLParser";
import { SelectColumnListContext } from "./TSQLParser";
import { SelectColumnContext } from "./TSQLParser";
import { FromClauseContext } from "./TSQLParser";
import { TableReferenceContext } from "./TSQLParser";


/**
 * This interface defines a complete listener for a parse tree produced by
 * `TSQLParser`.
 */
export interface TSQLListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by `TSQLParser.start`.
	 * @param ctx the parse tree
	 */
	enterStart?: (ctx: StartContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.start`.
	 * @param ctx the parse tree
	 */
	exitStart?: (ctx: StartContext) => void;

	/**
	 * Enter a parse tree produced by `TSQLParser.selectStatement`.
	 * @param ctx the parse tree
	 */
	enterSelectStatement?: (ctx: SelectStatementContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.selectStatement`.
	 * @param ctx the parse tree
	 */
	exitSelectStatement?: (ctx: SelectStatementContext) => void;

	/**
	 * Enter a parse tree produced by `TSQLParser.selectColumnList`.
	 * @param ctx the parse tree
	 */
	enterSelectColumnList?: (ctx: SelectColumnListContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.selectColumnList`.
	 * @param ctx the parse tree
	 */
	exitSelectColumnList?: (ctx: SelectColumnListContext) => void;

	/**
	 * Enter a parse tree produced by `TSQLParser.selectColumn`.
	 * @param ctx the parse tree
	 */
	enterSelectColumn?: (ctx: SelectColumnContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.selectColumn`.
	 * @param ctx the parse tree
	 */
	exitSelectColumn?: (ctx: SelectColumnContext) => void;

	/**
	 * Enter a parse tree produced by `TSQLParser.fromClause`.
	 * @param ctx the parse tree
	 */
	enterFromClause?: (ctx: FromClauseContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.fromClause`.
	 * @param ctx the parse tree
	 */
	exitFromClause?: (ctx: FromClauseContext) => void;

	/**
	 * Enter a parse tree produced by `TSQLParser.tableReference`.
	 * @param ctx the parse tree
	 */
	enterTableReference?: (ctx: TableReferenceContext) => void;
	/**
	 * Exit a parse tree produced by `TSQLParser.tableReference`.
	 * @param ctx the parse tree
	 */
	exitTableReference?: (ctx: TableReferenceContext) => void;
}

