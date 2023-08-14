// Generated from ./src/antlr/TSQL.g4 by ANTLR 4.9.0-SNAPSHOT


import { ATN } from "antlr4ts/atn/ATN";
import { ATNDeserializer } from "antlr4ts/atn/ATNDeserializer";
import { CharStream } from "antlr4ts/CharStream";
import { Lexer } from "antlr4ts/Lexer";
import { LexerATNSimulator } from "antlr4ts/atn/LexerATNSimulator";
import { NotNull } from "antlr4ts/Decorators";
import { Override } from "antlr4ts/Decorators";
import { RuleContext } from "antlr4ts/RuleContext";
import { Vocabulary } from "antlr4ts/Vocabulary";
import { VocabularyImpl } from "antlr4ts/VocabularyImpl";

import * as Utils from "antlr4ts/misc/Utils";


export class TSQLLexer extends Lexer {
	public static readonly T__0 = 1;
	public static readonly T__1 = 2;
	public static readonly T__2 = 3;
	public static readonly SELECT = 4;
	public static readonly FROM = 5;
	public static readonly AS = 6;
	public static readonly WS = 7;
	public static readonly ID = 8;

	// tslint:disable:no-trailing-whitespace
	public static readonly channelNames: string[] = [
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN",
	];

	// tslint:disable:no-trailing-whitespace
	public static readonly modeNames: string[] = [
		"DEFAULT_MODE",
	];

	public static readonly ruleNames: string[] = [
		"T__0", "T__1", "T__2", "SELECT", "FROM", "AS", "WS", "ID",
	];

	private static readonly _LITERAL_NAMES: Array<string | undefined> = [
		undefined, "','", "'('", "')'",
	];
	private static readonly _SYMBOLIC_NAMES: Array<string | undefined> = [
		undefined, undefined, undefined, undefined, "SELECT", "FROM", "AS", "WS", 
		"ID",
	];
	public static readonly VOCABULARY: Vocabulary = new VocabularyImpl(TSQLLexer._LITERAL_NAMES, TSQLLexer._SYMBOLIC_NAMES, []);

	// @Override
	// @NotNull
	public get vocabulary(): Vocabulary {
		return TSQLLexer.VOCABULARY;
	}
	// tslint:enable:no-trailing-whitespace


	constructor(input: CharStream) {
		super(input);
		this._interp = new LexerATNSimulator(TSQLLexer._ATN, this);
	}

	// @Override
	public get grammarFileName(): string { return "TSQL.g4"; }

	// @Override
	public get ruleNames(): string[] { return TSQLLexer.ruleNames; }

	// @Override
	public get serializedATN(): string { return TSQLLexer._serializedATN; }

	// @Override
	public get channelNames(): string[] { return TSQLLexer.channelNames; }

	// @Override
	public get modeNames(): string[] { return TSQLLexer.modeNames; }

	public static readonly _serializedATN: string =
		"\x03\uC91D\uCABA\u058D\uAFBA\u4F53\u0607\uEA8B\uC241\x02\n6\b\x01\x04" +
		"\x02\t\x02\x04\x03\t\x03\x04\x04\t\x04\x04\x05\t\x05\x04\x06\t\x06\x04" +
		"\x07\t\x07\x04\b\t\b\x04\t\t\t\x03\x02\x03\x02\x03\x03\x03\x03\x03\x04" +
		"\x03\x04\x03\x05\x03\x05\x03\x05\x03\x05\x03\x05\x03\x05\x03\x05\x03\x06" +
		"\x03\x06\x03\x06\x03\x06\x03\x06\x03\x07\x03\x07\x03\x07\x03\b\x06\b*" +
		"\n\b\r\b\x0E\b+\x03\b\x03\b\x03\t\x03\t\x07\t2\n\t\f\t\x0E\t5\v\t\x02" +
		"\x02\x02\n\x03\x02\x03\x05\x02\x04\x07\x02\x05\t\x02\x06\v\x02\x07\r\x02" +
		"\b\x0F\x02\t\x11\x02\n\x03\x02\x0F\x04\x02UUuu\x04\x02GGgg\x04\x02NNn" +
		"n\x04\x02EEee\x04\x02VVvv\x04\x02HHhh\x04\x02TTtt\x04\x02QQqq\x04\x02" +
		"OOoo\x04\x02CCcc\x05\x02\v\f\x0F\x0F\"\"\x05\x02C\\aac|\x06\x022;C\\a" +
		"ac|\x027\x02\x03\x03\x02\x02\x02\x02\x05\x03\x02\x02\x02\x02\x07\x03\x02" +
		"\x02\x02\x02\t\x03\x02\x02\x02\x02\v\x03\x02\x02\x02\x02\r\x03\x02\x02" +
		"\x02\x02\x0F\x03\x02\x02\x02\x02\x11\x03\x02\x02\x02\x03\x13\x03\x02\x02" +
		"\x02\x05\x15\x03\x02\x02\x02\x07\x17\x03\x02\x02\x02\t\x19\x03\x02\x02" +
		"\x02\v \x03\x02\x02\x02\r%\x03\x02\x02\x02\x0F)\x03\x02\x02\x02\x11/\x03" +
		"\x02\x02\x02\x13\x14\x07.\x02\x02\x14\x04\x03\x02\x02\x02\x15\x16\x07" +
		"*\x02\x02\x16\x06\x03\x02\x02\x02\x17\x18\x07+\x02\x02\x18\b\x03\x02\x02" +
		"\x02\x19\x1A\t\x02\x02\x02\x1A\x1B\t\x03\x02\x02\x1B\x1C\t\x04\x02\x02" +
		"\x1C\x1D\t\x03\x02\x02\x1D\x1E\t\x05\x02\x02\x1E\x1F\t\x06\x02\x02\x1F" +
		"\n\x03\x02\x02\x02 !\t\x07\x02\x02!\"\t\b\x02\x02\"#\t\t\x02\x02#$\t\n" +
		"\x02\x02$\f\x03\x02\x02\x02%&\t\v\x02\x02&\'\t\x02\x02\x02\'\x0E\x03\x02" +
		"\x02\x02(*\t\f\x02\x02)(\x03\x02\x02\x02*+\x03\x02\x02\x02+)\x03\x02\x02" +
		"\x02+,\x03\x02\x02\x02,-\x03\x02\x02\x02-.\b\b\x02\x02.\x10\x03\x02\x02" +
		"\x02/3\t\r\x02\x0202\t\x0E\x02\x0210\x03\x02\x02\x0225\x03\x02\x02\x02" +
		"31\x03\x02\x02\x0234\x03\x02\x02\x024\x12\x03\x02\x02\x0253\x03\x02\x02" +
		"\x02\x05\x02+3\x03\b\x02\x02";
	public static __ATN: ATN;
	public static get _ATN(): ATN {
		if (!TSQLLexer.__ATN) {
			TSQLLexer.__ATN = new ATNDeserializer().deserialize(Utils.toCharArray(TSQLLexer._serializedATN));
		}

		return TSQLLexer.__ATN;
	}

}

