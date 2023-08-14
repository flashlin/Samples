// Generated from ./src/antlr/TSQL.g4 by ANTLR 4.9.0-SNAPSHOT


import { ATN } from "antlr4ts/atn/ATN";
import { ATNDeserializer } from "antlr4ts/atn/ATNDeserializer";
import { FailedPredicateException } from "antlr4ts/FailedPredicateException";
import { NotNull } from "antlr4ts/Decorators";
import { NoViableAltException } from "antlr4ts/NoViableAltException";
import { Override } from "antlr4ts/Decorators";
import { Parser } from "antlr4ts/Parser";
import { ParserRuleContext } from "antlr4ts/ParserRuleContext";
import { ParserATNSimulator } from "antlr4ts/atn/ParserATNSimulator";
import { ParseTreeListener } from "antlr4ts/tree/ParseTreeListener";
import { ParseTreeVisitor } from "antlr4ts/tree/ParseTreeVisitor";
import { RecognitionException } from "antlr4ts/RecognitionException";
import { RuleContext } from "antlr4ts/RuleContext";
//import { RuleVersion } from "antlr4ts/RuleVersion";
import { TerminalNode } from "antlr4ts/tree/TerminalNode";
import { Token } from "antlr4ts/Token";
import { TokenStream } from "antlr4ts/TokenStream";
import { Vocabulary } from "antlr4ts/Vocabulary";
import { VocabularyImpl } from "antlr4ts/VocabularyImpl";

import * as Utils from "antlr4ts/misc/Utils";

import { TSQLListener } from "./TSQLListener";
import { TSQLVisitor } from "./TSQLVisitor";


export class TSQLParser extends Parser {
	public static readonly T__0 = 1;
	public static readonly T__1 = 2;
	public static readonly T__2 = 3;
	public static readonly SELECT = 4;
	public static readonly FROM = 5;
	public static readonly AS = 6;
	public static readonly WS = 7;
	public static readonly ID = 8;
	public static readonly RULE_start = 0;
	public static readonly RULE_selectStatement = 1;
	public static readonly RULE_selectColumnList = 2;
	public static readonly RULE_selectColumn = 3;
	public static readonly RULE_fromClause = 4;
	public static readonly RULE_tableReference = 5;
	// tslint:disable:no-trailing-whitespace
	public static readonly ruleNames: string[] = [
		"start", "selectStatement", "selectColumnList", "selectColumn", "fromClause", 
		"tableReference",
	];

	private static readonly _LITERAL_NAMES: Array<string | undefined> = [
		undefined, "','", "'('", "')'",
	];
	private static readonly _SYMBOLIC_NAMES: Array<string | undefined> = [
		undefined, undefined, undefined, undefined, "SELECT", "FROM", "AS", "WS", 
		"ID",
	];
	public static readonly VOCABULARY: Vocabulary = new VocabularyImpl(TSQLParser._LITERAL_NAMES, TSQLParser._SYMBOLIC_NAMES, []);

	// @Override
	// @NotNull
	public get vocabulary(): Vocabulary {
		return TSQLParser.VOCABULARY;
	}
	// tslint:enable:no-trailing-whitespace

	// @Override
	public get grammarFileName(): string { return "TSQL.g4"; }

	// @Override
	public get ruleNames(): string[] { return TSQLParser.ruleNames; }

	// @Override
	public get serializedATN(): string { return TSQLParser._serializedATN; }

	protected createFailedPredicateException(predicate?: string, message?: string): FailedPredicateException {
		return new FailedPredicateException(this, predicate, message);
	}

	constructor(input: TokenStream) {
		super(input);
		this._interp = new ParserATNSimulator(TSQLParser._ATN, this);
	}
	// @RuleVersion(0)
	public start(): StartContext {
		let _localctx: StartContext = new StartContext(this._ctx, this.state);
		this.enterRule(_localctx, 0, TSQLParser.RULE_start);
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 12;
			this.selectStatement();
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public selectStatement(): SelectStatementContext {
		let _localctx: SelectStatementContext = new SelectStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 2, TSQLParser.RULE_selectStatement);
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 14;
			this.match(TSQLParser.SELECT);
			this.state = 15;
			this.selectColumnList();
			this.state = 16;
			this.match(TSQLParser.FROM);
			this.state = 17;
			this.fromClause();
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public selectColumnList(): SelectColumnListContext {
		let _localctx: SelectColumnListContext = new SelectColumnListContext(this._ctx, this.state);
		this.enterRule(_localctx, 4, TSQLParser.RULE_selectColumnList);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 19;
			this.selectColumn();
			this.state = 24;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			while (_la === TSQLParser.T__0) {
				{
				{
				this.state = 20;
				this.match(TSQLParser.T__0);
				this.state = 21;
				this.selectColumn();
				}
				}
				this.state = 26;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public selectColumn(): SelectColumnContext {
		let _localctx: SelectColumnContext = new SelectColumnContext(this._ctx, this.state);
		this.enterRule(_localctx, 6, TSQLParser.RULE_selectColumn);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 27;
			this.match(TSQLParser.ID);
			this.state = 32;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === TSQLParser.AS || _la === TSQLParser.ID) {
				{
				this.state = 29;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === TSQLParser.AS) {
					{
					this.state = 28;
					this.match(TSQLParser.AS);
					}
				}

				this.state = 31;
				this.match(TSQLParser.ID);
				}
			}

			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public fromClause(): FromClauseContext {
		let _localctx: FromClauseContext = new FromClauseContext(this._ctx, this.state);
		this.enterRule(_localctx, 8, TSQLParser.RULE_fromClause);
		try {
			this.state = 39;
			this._errHandler.sync(this);
			switch (this._input.LA(1)) {
			case TSQLParser.ID:
				this.enterOuterAlt(_localctx, 1);
				{
				this.state = 34;
				this.tableReference();
				}
				break;
			case TSQLParser.T__1:
				this.enterOuterAlt(_localctx, 2);
				{
				this.state = 35;
				this.match(TSQLParser.T__1);
				this.state = 36;
				this.selectStatement();
				this.state = 37;
				this.match(TSQLParser.T__2);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public tableReference(): TableReferenceContext {
		let _localctx: TableReferenceContext = new TableReferenceContext(this._ctx, this.state);
		this.enterRule(_localctx, 10, TSQLParser.RULE_tableReference);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 41;
			this.match(TSQLParser.ID);
			this.state = 46;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === TSQLParser.AS || _la === TSQLParser.ID) {
				{
				this.state = 43;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === TSQLParser.AS) {
					{
					this.state = 42;
					this.match(TSQLParser.AS);
					}
				}

				this.state = 45;
				this.match(TSQLParser.ID);
				}
			}

			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}

	public static readonly _serializedATN: string =
		"\x03\uC91D\uCABA\u058D\uAFBA\u4F53\u0607\uEA8B\uC241\x03\n3\x04\x02\t" +
		"\x02\x04\x03\t\x03\x04\x04\t\x04\x04\x05\t\x05\x04\x06\t\x06\x04\x07\t" +
		"\x07\x03\x02\x03\x02\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x04\x03" +
		"\x04\x03\x04\x07\x04\x19\n\x04\f\x04\x0E\x04\x1C\v\x04\x03\x05\x03\x05" +
		"\x05\x05 \n\x05\x03\x05\x05\x05#\n\x05\x03\x06\x03\x06\x03\x06\x03\x06" +
		"\x03\x06\x05\x06*\n\x06\x03\x07\x03\x07\x05\x07.\n\x07\x03\x07\x05\x07" +
		"1\n\x07\x03\x07\x02\x02\x02\b\x02\x02\x04\x02\x06\x02\b\x02\n\x02\f\x02" +
		"\x02\x02\x022\x02\x0E\x03\x02\x02\x02\x04\x10\x03\x02\x02\x02\x06\x15" +
		"\x03\x02\x02\x02\b\x1D\x03\x02\x02\x02\n)\x03\x02\x02\x02\f+\x03\x02\x02" +
		"\x02\x0E\x0F\x05\x04\x03\x02\x0F\x03\x03\x02\x02\x02\x10\x11\x07\x06\x02" +
		"\x02\x11\x12\x05\x06\x04\x02\x12\x13\x07\x07\x02\x02\x13\x14\x05\n\x06" +
		"\x02\x14\x05\x03\x02\x02\x02\x15\x1A\x05\b\x05\x02\x16\x17\x07\x03\x02" +
		"\x02\x17\x19\x05\b\x05\x02\x18\x16\x03\x02\x02\x02\x19\x1C\x03\x02\x02" +
		"\x02\x1A\x18\x03\x02\x02\x02\x1A\x1B\x03\x02\x02\x02\x1B\x07\x03\x02\x02" +
		"\x02\x1C\x1A\x03\x02\x02\x02\x1D\"\x07\n\x02\x02\x1E \x07\b\x02\x02\x1F" +
		"\x1E\x03\x02\x02\x02\x1F \x03\x02\x02\x02 !\x03\x02\x02\x02!#\x07\n\x02" +
		"\x02\"\x1F\x03\x02\x02\x02\"#\x03\x02\x02\x02#\t\x03\x02\x02\x02$*\x05" +
		"\f\x07\x02%&\x07\x04\x02\x02&\'\x05\x04\x03\x02\'(\x07\x05\x02\x02(*\x03" +
		"\x02\x02\x02)$\x03\x02\x02\x02)%\x03\x02\x02\x02*\v\x03\x02\x02\x02+0" +
		"\x07\n\x02\x02,.\x07\b\x02\x02-,\x03\x02\x02\x02-.\x03\x02\x02\x02./\x03" +
		"\x02\x02\x02/1\x07\n\x02\x020-\x03\x02\x02\x0201\x03\x02\x02\x021\r\x03" +
		"\x02\x02\x02\b\x1A\x1F\")-0";
	public static __ATN: ATN;
	public static get _ATN(): ATN {
		if (!TSQLParser.__ATN) {
			TSQLParser.__ATN = new ATNDeserializer().deserialize(Utils.toCharArray(TSQLParser._serializedATN));
		}

		return TSQLParser.__ATN;
	}

}

export class StartContext extends ParserRuleContext {
	public selectStatement(): SelectStatementContext {
		return this.getRuleContext(0, SelectStatementContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_start; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterStart) {
			listener.enterStart(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitStart) {
			listener.exitStart(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitStart) {
			return visitor.visitStart(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class SelectStatementContext extends ParserRuleContext {
	public SELECT(): TerminalNode { return this.getToken(TSQLParser.SELECT, 0); }
	public selectColumnList(): SelectColumnListContext {
		return this.getRuleContext(0, SelectColumnListContext);
	}
	public FROM(): TerminalNode { return this.getToken(TSQLParser.FROM, 0); }
	public fromClause(): FromClauseContext {
		return this.getRuleContext(0, FromClauseContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_selectStatement; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterSelectStatement) {
			listener.enterSelectStatement(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitSelectStatement) {
			listener.exitSelectStatement(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitSelectStatement) {
			return visitor.visitSelectStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class SelectColumnListContext extends ParserRuleContext {
	public selectColumn(): SelectColumnContext[];
	public selectColumn(i: number): SelectColumnContext;
	public selectColumn(i?: number): SelectColumnContext | SelectColumnContext[] {
		if (i === undefined) {
			return this.getRuleContexts(SelectColumnContext);
		} else {
			return this.getRuleContext(i, SelectColumnContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_selectColumnList; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterSelectColumnList) {
			listener.enterSelectColumnList(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitSelectColumnList) {
			listener.exitSelectColumnList(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitSelectColumnList) {
			return visitor.visitSelectColumnList(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class SelectColumnContext extends ParserRuleContext {
	public ID(): TerminalNode[];
	public ID(i: number): TerminalNode;
	public ID(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(TSQLParser.ID);
		} else {
			return this.getToken(TSQLParser.ID, i);
		}
	}
	public AS(): TerminalNode | undefined { return this.tryGetToken(TSQLParser.AS, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_selectColumn; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterSelectColumn) {
			listener.enterSelectColumn(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitSelectColumn) {
			listener.exitSelectColumn(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitSelectColumn) {
			return visitor.visitSelectColumn(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FromClauseContext extends ParserRuleContext {
	public tableReference(): TableReferenceContext | undefined {
		return this.tryGetRuleContext(0, TableReferenceContext);
	}
	public selectStatement(): SelectStatementContext | undefined {
		return this.tryGetRuleContext(0, SelectStatementContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_fromClause; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterFromClause) {
			listener.enterFromClause(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitFromClause) {
			listener.exitFromClause(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitFromClause) {
			return visitor.visitFromClause(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class TableReferenceContext extends ParserRuleContext {
	public ID(): TerminalNode[];
	public ID(i: number): TerminalNode;
	public ID(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(TSQLParser.ID);
		} else {
			return this.getToken(TSQLParser.ID, i);
		}
	}
	public AS(): TerminalNode | undefined { return this.tryGetToken(TSQLParser.AS, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return TSQLParser.RULE_tableReference; }
	// @Override
	public enterRule(listener: TSQLListener): void {
		if (listener.enterTableReference) {
			listener.enterTableReference(this);
		}
	}
	// @Override
	public exitRule(listener: TSQLListener): void {
		if (listener.exitTableReference) {
			listener.exitTableReference(this);
		}
	}
	// @Override
	public accept<Result>(visitor: TSQLVisitor<Result>): Result {
		if (visitor.visitTableReference) {
			return visitor.visitTableReference(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


