//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     ANTLR Version: 4.7.2
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

// Generated from TsqlLexer.g4 by ANTLR 4.7.2

// Unreachable code detected
#pragma warning disable 0162
// The variable '...' is assigned but its value is never used
#pragma warning disable 0219
// Missing XML comment for publicly visible type or member '...'
#pragma warning disable 1591
// Ambiguous reference in cref attribute
#pragma warning disable 419

namespace T1.SqlDom.Tsql {
using System;
using System.IO;
using System.Text;
using Antlr4.Runtime;
using Antlr4.Runtime.Atn;
using Antlr4.Runtime.Misc;
using DFA = Antlr4.Runtime.Dfa.DFA;

[System.CodeDom.Compiler.GeneratedCode("ANTLR", "4.7.2")]
[System.CLSCompliant(false)]
public partial class TsqlLexer : Lexer {
	protected static DFA[] decisionToDFA;
	protected static PredictionContextCache sharedContextCache = new PredictionContextCache();
	public const int
		SPACE=1, SPEC_ESSQL_COMMENT=2, COMMENT_INPUT=3, LINE_COMMENT=4, DOT=5, 
		UNDERLINE=6, LBRACKET=7, RBRACKET=8, LPAREN=9, RPAREN=10, MINUS=11, STAR=12, 
		COMMA=13, SEMI=14, GT=15, SINGLE_QUOTE=16, DOUBLE_QUOTE=17, REVERSE_QUOTE=18, 
		COLON=19, EQ=20, NE=21, BOOLOR=22, BOOLAND=23, INT=24, DECIMAL=25, SELECT=26, 
		ALL=27, DISTINCT=28, DELETED=29, INSERTED=30, TOP=31, AS=32, ID_LETTER=33, 
		SQUARE_BRACKET_ID=34, STRING=35;
	public const int
		ESQLCOMMENT=2;
	public static string[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN", "ESQLCOMMENT"
	};

	public static string[] modeNames = {
		"DEFAULT_MODE"
	};

	public static readonly string[] ruleNames = {
		"SPACE", "SPEC_ESSQL_COMMENT", "COMMENT_INPUT", "LINE_COMMENT", "DOT", 
		"UNDERLINE", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", "MINUS", "STAR", 
		"COMMA", "SEMI", "GT", "SINGLE_QUOTE", "DOUBLE_QUOTE", "REVERSE_QUOTE", 
		"COLON", "EQ", "NE", "BOOLOR", "BOOLAND", "INT", "DECIMAL", "SELECT", 
		"ALL", "DISTINCT", "DELETED", "INSERTED", "TOP", "AS", "ID_LETTER", "SQUARE_BRACKET_ID", 
		"STRING", "DEC_DIGIT", "LETTER", "A", "B", "C", "D", "E", "F", "G", "H", 
		"I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", 
		"W", "X", "Y", "Z"
	};


	public TsqlLexer(ICharStream input)
	: this(input, Console.Out, Console.Error) { }

	public TsqlLexer(ICharStream input, TextWriter output, TextWriter errorOutput)
	: base(input, output, errorOutput)
	{
		Interpreter = new LexerATNSimulator(this, _ATN, decisionToDFA, sharedContextCache);
	}

	private static readonly string[] _LiteralNames = {
		null, null, null, null, null, "'.'", "'_'", "'['", "']'", "'('", "')'", 
		"'-'", "'*'", null, "';'", "'>'", "'''", "'\"'", "'`'", null, "'='", "'!='"
	};
	private static readonly string[] _SymbolicNames = {
		null, "SPACE", "SPEC_ESSQL_COMMENT", "COMMENT_INPUT", "LINE_COMMENT", 
		"DOT", "UNDERLINE", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", "MINUS", 
		"STAR", "COMMA", "SEMI", "GT", "SINGLE_QUOTE", "DOUBLE_QUOTE", "REVERSE_QUOTE", 
		"COLON", "EQ", "NE", "BOOLOR", "BOOLAND", "INT", "DECIMAL", "SELECT", 
		"ALL", "DISTINCT", "DELETED", "INSERTED", "TOP", "AS", "ID_LETTER", "SQUARE_BRACKET_ID", 
		"STRING"
	};
	public static readonly IVocabulary DefaultVocabulary = new Vocabulary(_LiteralNames, _SymbolicNames);

	[NotNull]
	public override IVocabulary Vocabulary
	{
		get
		{
			return DefaultVocabulary;
		}
	}

	public override string GrammarFileName { get { return "TsqlLexer.g4"; } }

	public override string[] RuleNames { get { return ruleNames; } }

	public override string[] ChannelNames { get { return channelNames; } }

	public override string[] ModeNames { get { return modeNames; } }

	public override string SerializedAtn { get { return new string(_serializedATN); } }

	static TsqlLexer() {
		decisionToDFA = new DFA[_ATN.NumberOfDecisions];
		for (int i = 0; i < _ATN.NumberOfDecisions; i++) {
			decisionToDFA[i] = new DFA(_ATN.GetDecisionState(i), i);
		}
	}
	private static char[] _serializedATN = {
		'\x3', '\x608B', '\xA72A', '\x8133', '\xB9ED', '\x417C', '\x3BE7', '\x7786', 
		'\x5964', '\x2', '%', '\x199', '\b', '\x1', '\x4', '\x2', '\t', '\x2', 
		'\x4', '\x3', '\t', '\x3', '\x4', '\x4', '\t', '\x4', '\x4', '\x5', '\t', 
		'\x5', '\x4', '\x6', '\t', '\x6', '\x4', '\a', '\t', '\a', '\x4', '\b', 
		'\t', '\b', '\x4', '\t', '\t', '\t', '\x4', '\n', '\t', '\n', '\x4', '\v', 
		'\t', '\v', '\x4', '\f', '\t', '\f', '\x4', '\r', '\t', '\r', '\x4', '\xE', 
		'\t', '\xE', '\x4', '\xF', '\t', '\xF', '\x4', '\x10', '\t', '\x10', '\x4', 
		'\x11', '\t', '\x11', '\x4', '\x12', '\t', '\x12', '\x4', '\x13', '\t', 
		'\x13', '\x4', '\x14', '\t', '\x14', '\x4', '\x15', '\t', '\x15', '\x4', 
		'\x16', '\t', '\x16', '\x4', '\x17', '\t', '\x17', '\x4', '\x18', '\t', 
		'\x18', '\x4', '\x19', '\t', '\x19', '\x4', '\x1A', '\t', '\x1A', '\x4', 
		'\x1B', '\t', '\x1B', '\x4', '\x1C', '\t', '\x1C', '\x4', '\x1D', '\t', 
		'\x1D', '\x4', '\x1E', '\t', '\x1E', '\x4', '\x1F', '\t', '\x1F', '\x4', 
		' ', '\t', ' ', '\x4', '!', '\t', '!', '\x4', '\"', '\t', '\"', '\x4', 
		'#', '\t', '#', '\x4', '$', '\t', '$', '\x4', '%', '\t', '%', '\x4', '&', 
		'\t', '&', '\x4', '\'', '\t', '\'', '\x4', '(', '\t', '(', '\x4', ')', 
		'\t', ')', '\x4', '*', '\t', '*', '\x4', '+', '\t', '+', '\x4', ',', '\t', 
		',', '\x4', '-', '\t', '-', '\x4', '.', '\t', '.', '\x4', '/', '\t', '/', 
		'\x4', '\x30', '\t', '\x30', '\x4', '\x31', '\t', '\x31', '\x4', '\x32', 
		'\t', '\x32', '\x4', '\x33', '\t', '\x33', '\x4', '\x34', '\t', '\x34', 
		'\x4', '\x35', '\t', '\x35', '\x4', '\x36', '\t', '\x36', '\x4', '\x37', 
		'\t', '\x37', '\x4', '\x38', '\t', '\x38', '\x4', '\x39', '\t', '\x39', 
		'\x4', ':', '\t', ':', '\x4', ';', '\t', ';', '\x4', '<', '\t', '<', '\x4', 
		'=', '\t', '=', '\x4', '>', '\t', '>', '\x4', '?', '\t', '?', '\x4', '@', 
		'\t', '@', '\x3', '\x2', '\x6', '\x2', '\x83', '\n', '\x2', '\r', '\x2', 
		'\xE', '\x2', '\x84', '\x3', '\x2', '\x3', '\x2', '\x3', '\x3', '\x3', 
		'\x3', '\x3', '\x3', '\x3', '\x3', '\x3', '\x3', '\x6', '\x3', '\x8E', 
		'\n', '\x3', '\r', '\x3', '\xE', '\x3', '\x8F', '\x3', '\x3', '\x3', '\x3', 
		'\x3', '\x3', '\x3', '\x3', '\x3', '\x3', '\x3', '\x4', '\x3', '\x4', 
		'\x3', '\x4', '\x3', '\x4', '\a', '\x4', '\x9B', '\n', '\x4', '\f', '\x4', 
		'\xE', '\x4', '\x9E', '\v', '\x4', '\x3', '\x4', '\x3', '\x4', '\x3', 
		'\x4', '\x3', '\x4', '\x3', '\x4', '\x3', '\x5', '\x3', '\x5', '\x3', 
		'\x5', '\x3', '\x5', '\x5', '\x5', '\xA9', '\n', '\x5', '\x3', '\x5', 
		'\a', '\x5', '\xAC', '\n', '\x5', '\f', '\x5', '\xE', '\x5', '\xAF', '\v', 
		'\x5', '\x3', '\x5', '\x5', '\x5', '\xB2', '\n', '\x5', '\x3', '\x5', 
		'\x3', '\x5', '\x5', '\x5', '\xB6', '\n', '\x5', '\x3', '\x5', '\x3', 
		'\x5', '\x3', '\x5', '\x3', '\x5', '\x5', '\x5', '\xBC', '\n', '\x5', 
		'\x3', '\x5', '\x3', '\x5', '\x5', '\x5', '\xC0', '\n', '\x5', '\x5', 
		'\x5', '\xC2', '\n', '\x5', '\x3', '\x5', '\x3', '\x5', '\x3', '\x6', 
		'\x3', '\x6', '\x3', '\a', '\x3', '\a', '\x3', '\b', '\x3', '\b', '\x3', 
		'\t', '\x3', '\t', '\x3', '\n', '\x3', '\n', '\x3', '\v', '\x3', '\v', 
		'\x3', '\f', '\x3', '\f', '\x3', '\r', '\x3', '\r', '\x3', '\xE', '\x3', 
		'\xE', '\x3', '\xF', '\x3', '\xF', '\x3', '\x10', '\x3', '\x10', '\x3', 
		'\x11', '\x3', '\x11', '\x3', '\x12', '\x3', '\x12', '\x3', '\x13', '\x3', 
		'\x13', '\x3', '\x14', '\x3', '\x14', '\x3', '\x15', '\x3', '\x15', '\x3', 
		'\x16', '\x3', '\x16', '\x3', '\x16', '\x3', '\x17', '\x3', '\x17', '\x3', 
		'\x17', '\x5', '\x17', '\xEC', '\n', '\x17', '\x3', '\x18', '\x3', '\x18', 
		'\x3', '\x18', '\x3', '\x18', '\x5', '\x18', '\xF2', '\n', '\x18', '\x3', 
		'\x19', '\x5', '\x19', '\xF5', '\n', '\x19', '\x3', '\x19', '\x6', '\x19', 
		'\xF8', '\n', '\x19', '\r', '\x19', '\xE', '\x19', '\xF9', '\x3', '\x1A', 
		'\x5', '\x1A', '\xFD', '\n', '\x1A', '\x3', '\x1A', '\x6', '\x1A', '\x100', 
		'\n', '\x1A', '\r', '\x1A', '\xE', '\x1A', '\x101', '\x3', '\x1A', '\x3', 
		'\x1A', '\x6', '\x1A', '\x106', '\n', '\x1A', '\r', '\x1A', '\xE', '\x1A', 
		'\x107', '\x3', '\x1A', '\x5', '\x1A', '\x10B', '\n', '\x1A', '\x3', '\x1A', 
		'\x3', '\x1A', '\x6', '\x1A', '\x10F', '\n', '\x1A', '\r', '\x1A', '\xE', 
		'\x1A', '\x110', '\x5', '\x1A', '\x113', '\n', '\x1A', '\x3', '\x1B', 
		'\x3', '\x1B', '\x3', '\x1B', '\x3', '\x1B', '\x3', '\x1B', '\x3', '\x1B', 
		'\x3', '\x1B', '\x3', '\x1C', '\x3', '\x1C', '\x3', '\x1C', '\x3', '\x1C', 
		'\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1D', 
		'\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1D', '\x3', '\x1E', 
		'\x3', '\x1E', '\x3', '\x1E', '\x3', '\x1E', '\x3', '\x1E', '\x3', '\x1E', 
		'\x3', '\x1E', '\x3', '\x1E', '\x3', '\x1F', '\x3', '\x1F', '\x3', '\x1F', 
		'\x3', '\x1F', '\x3', '\x1F', '\x3', '\x1F', '\x3', '\x1F', '\x3', '\x1F', 
		'\x3', '\x1F', '\x3', ' ', '\x3', ' ', '\x3', ' ', '\x3', ' ', '\x3', 
		'!', '\x3', '!', '\x3', '!', '\x3', '\"', '\x6', '\"', '\x142', '\n', 
		'\"', '\r', '\"', '\xE', '\"', '\x143', '\x3', '\"', '\x3', '\"', '\x3', 
		'\"', '\x6', '\"', '\x149', '\n', '\"', '\r', '\"', '\xE', '\"', '\x14A', 
		'\x5', '\"', '\x14D', '\n', '\"', '\x3', '#', '\x3', '#', '\x6', '#', 
		'\x151', '\n', '#', '\r', '#', '\xE', '#', '\x152', '\x3', '#', '\x3', 
		'#', '\x3', '$', '\x3', '$', '\a', '$', '\x159', '\n', '$', '\f', '$', 
		'\xE', '$', '\x15C', '\v', '$', '\x3', '$', '\x3', '$', '\x3', '%', '\x3', 
		'%', '\x3', '&', '\x3', '&', '\x5', '&', '\x164', '\n', '&', '\x3', '\'', 
		'\x3', '\'', '\x3', '(', '\x3', '(', '\x3', ')', '\x3', ')', '\x3', '*', 
		'\x3', '*', '\x3', '+', '\x3', '+', '\x3', ',', '\x3', ',', '\x3', '-', 
		'\x3', '-', '\x3', '.', '\x3', '.', '\x3', '/', '\x3', '/', '\x3', '\x30', 
		'\x3', '\x30', '\x3', '\x31', '\x3', '\x31', '\x3', '\x32', '\x3', '\x32', 
		'\x3', '\x33', '\x3', '\x33', '\x3', '\x34', '\x3', '\x34', '\x3', '\x35', 
		'\x3', '\x35', '\x3', '\x36', '\x3', '\x36', '\x3', '\x37', '\x3', '\x37', 
		'\x3', '\x38', '\x3', '\x38', '\x3', '\x39', '\x3', '\x39', '\x3', ':', 
		'\x3', ':', '\x3', ';', '\x3', ';', '\x3', '<', '\x3', '<', '\x3', '=', 
		'\x3', '=', '\x3', '>', '\x3', '>', '\x3', '?', '\x3', '?', '\x3', '@', 
		'\x3', '@', '\x4', '\x8F', '\x9C', '\x2', '\x41', '\x3', '\x3', '\x5', 
		'\x4', '\a', '\x5', '\t', '\x6', '\v', '\a', '\r', '\b', '\xF', '\t', 
		'\x11', '\n', '\x13', '\v', '\x15', '\f', '\x17', '\r', '\x19', '\xE', 
		'\x1B', '\xF', '\x1D', '\x10', '\x1F', '\x11', '!', '\x12', '#', '\x13', 
		'%', '\x14', '\'', '\x15', ')', '\x16', '+', '\x17', '-', '\x18', '/', 
		'\x19', '\x31', '\x1A', '\x33', '\x1B', '\x35', '\x1C', '\x37', '\x1D', 
		'\x39', '\x1E', ';', '\x1F', '=', ' ', '?', '!', '\x41', '\"', '\x43', 
		'#', '\x45', '$', 'G', '%', 'I', '\x2', 'K', '\x2', 'M', '\x2', 'O', '\x2', 
		'Q', '\x2', 'S', '\x2', 'U', '\x2', 'W', '\x2', 'Y', '\x2', '[', '\x2', 
		']', '\x2', '_', '\x2', '\x61', '\x2', '\x63', '\x2', '\x65', '\x2', 'g', 
		'\x2', 'i', '\x2', 'k', '\x2', 'm', '\x2', 'o', '\x2', 'q', '\x2', 's', 
		'\x2', 'u', '\x2', 'w', '\x2', 'y', '\x2', '{', '\x2', '}', '\x2', '\x7F', 
		'\x2', '\x3', '\x2', '#', '\x5', '\x2', '\v', '\f', '\xF', '\xF', '\"', 
		'\"', '\x4', '\x2', '\f', '\f', '\xF', '\xF', '\x4', '\x2', '.', '.', 
		'\xFF0E', '\xFF0E', '\x4', '\x2', '<', '<', '\xFF1C', '\xFF1C', '\x3', 
		'\x2', ')', ')', '\x3', '\x2', '\x32', ';', '\x4', '\x2', '\x43', '\\', 
		'\x63', '|', '\x4', '\x2', '\x43', '\x43', '\x63', '\x63', '\x4', '\x2', 
		'\x44', '\x44', '\x64', '\x64', '\x4', '\x2', '\x45', '\x45', '\x65', 
		'\x65', '\x4', '\x2', '\x46', '\x46', '\x66', '\x66', '\x4', '\x2', 'G', 
		'G', 'g', 'g', '\x4', '\x2', 'H', 'H', 'h', 'h', '\x4', '\x2', 'I', 'I', 
		'i', 'i', '\x4', '\x2', 'J', 'J', 'j', 'j', '\x4', '\x2', 'K', 'K', 'k', 
		'k', '\x4', '\x2', 'L', 'L', 'l', 'l', '\x4', '\x2', 'M', 'M', 'm', 'm', 
		'\x4', '\x2', 'N', 'N', 'n', 'n', '\x4', '\x2', 'O', 'O', 'o', 'o', '\x4', 
		'\x2', 'P', 'P', 'p', 'p', '\x4', '\x2', 'Q', 'Q', 'q', 'q', '\x4', '\x2', 
		'R', 'R', 'r', 'r', '\x4', '\x2', 'S', 'S', 's', 's', '\x4', '\x2', 'T', 
		'T', 't', 't', '\x4', '\x2', 'U', 'U', 'u', 'u', '\x4', '\x2', 'V', 'V', 
		'v', 'v', '\x4', '\x2', 'W', 'W', 'w', 'w', '\x4', '\x2', 'X', 'X', 'x', 
		'x', '\x4', '\x2', 'Y', 'Y', 'y', 'y', '\x4', '\x2', 'Z', 'Z', 'z', 'z', 
		'\x4', '\x2', '[', '[', '{', '{', '\x4', '\x2', '\\', '\\', '|', '|', 
		'\x2', '\x198', '\x2', '\x3', '\x3', '\x2', '\x2', '\x2', '\x2', '\x5', 
		'\x3', '\x2', '\x2', '\x2', '\x2', '\a', '\x3', '\x2', '\x2', '\x2', '\x2', 
		'\t', '\x3', '\x2', '\x2', '\x2', '\x2', '\v', '\x3', '\x2', '\x2', '\x2', 
		'\x2', '\r', '\x3', '\x2', '\x2', '\x2', '\x2', '\xF', '\x3', '\x2', '\x2', 
		'\x2', '\x2', '\x11', '\x3', '\x2', '\x2', '\x2', '\x2', '\x13', '\x3', 
		'\x2', '\x2', '\x2', '\x2', '\x15', '\x3', '\x2', '\x2', '\x2', '\x2', 
		'\x17', '\x3', '\x2', '\x2', '\x2', '\x2', '\x19', '\x3', '\x2', '\x2', 
		'\x2', '\x2', '\x1B', '\x3', '\x2', '\x2', '\x2', '\x2', '\x1D', '\x3', 
		'\x2', '\x2', '\x2', '\x2', '\x1F', '\x3', '\x2', '\x2', '\x2', '\x2', 
		'!', '\x3', '\x2', '\x2', '\x2', '\x2', '#', '\x3', '\x2', '\x2', '\x2', 
		'\x2', '%', '\x3', '\x2', '\x2', '\x2', '\x2', '\'', '\x3', '\x2', '\x2', 
		'\x2', '\x2', ')', '\x3', '\x2', '\x2', '\x2', '\x2', '+', '\x3', '\x2', 
		'\x2', '\x2', '\x2', '-', '\x3', '\x2', '\x2', '\x2', '\x2', '/', '\x3', 
		'\x2', '\x2', '\x2', '\x2', '\x31', '\x3', '\x2', '\x2', '\x2', '\x2', 
		'\x33', '\x3', '\x2', '\x2', '\x2', '\x2', '\x35', '\x3', '\x2', '\x2', 
		'\x2', '\x2', '\x37', '\x3', '\x2', '\x2', '\x2', '\x2', '\x39', '\x3', 
		'\x2', '\x2', '\x2', '\x2', ';', '\x3', '\x2', '\x2', '\x2', '\x2', '=', 
		'\x3', '\x2', '\x2', '\x2', '\x2', '?', '\x3', '\x2', '\x2', '\x2', '\x2', 
		'\x41', '\x3', '\x2', '\x2', '\x2', '\x2', '\x43', '\x3', '\x2', '\x2', 
		'\x2', '\x2', '\x45', '\x3', '\x2', '\x2', '\x2', '\x2', 'G', '\x3', '\x2', 
		'\x2', '\x2', '\x3', '\x82', '\x3', '\x2', '\x2', '\x2', '\x5', '\x88', 
		'\x3', '\x2', '\x2', '\x2', '\a', '\x96', '\x3', '\x2', '\x2', '\x2', 
		'\t', '\xC1', '\x3', '\x2', '\x2', '\x2', '\v', '\xC5', '\x3', '\x2', 
		'\x2', '\x2', '\r', '\xC7', '\x3', '\x2', '\x2', '\x2', '\xF', '\xC9', 
		'\x3', '\x2', '\x2', '\x2', '\x11', '\xCB', '\x3', '\x2', '\x2', '\x2', 
		'\x13', '\xCD', '\x3', '\x2', '\x2', '\x2', '\x15', '\xCF', '\x3', '\x2', 
		'\x2', '\x2', '\x17', '\xD1', '\x3', '\x2', '\x2', '\x2', '\x19', '\xD3', 
		'\x3', '\x2', '\x2', '\x2', '\x1B', '\xD5', '\x3', '\x2', '\x2', '\x2', 
		'\x1D', '\xD7', '\x3', '\x2', '\x2', '\x2', '\x1F', '\xD9', '\x3', '\x2', 
		'\x2', '\x2', '!', '\xDB', '\x3', '\x2', '\x2', '\x2', '#', '\xDD', '\x3', 
		'\x2', '\x2', '\x2', '%', '\xDF', '\x3', '\x2', '\x2', '\x2', '\'', '\xE1', 
		'\x3', '\x2', '\x2', '\x2', ')', '\xE3', '\x3', '\x2', '\x2', '\x2', '+', 
		'\xE5', '\x3', '\x2', '\x2', '\x2', '-', '\xEB', '\x3', '\x2', '\x2', 
		'\x2', '/', '\xF1', '\x3', '\x2', '\x2', '\x2', '\x31', '\xF4', '\x3', 
		'\x2', '\x2', '\x2', '\x33', '\x112', '\x3', '\x2', '\x2', '\x2', '\x35', 
		'\x114', '\x3', '\x2', '\x2', '\x2', '\x37', '\x11B', '\x3', '\x2', '\x2', 
		'\x2', '\x39', '\x11F', '\x3', '\x2', '\x2', '\x2', ';', '\x128', '\x3', 
		'\x2', '\x2', '\x2', '=', '\x130', '\x3', '\x2', '\x2', '\x2', '?', '\x139', 
		'\x3', '\x2', '\x2', '\x2', '\x41', '\x13D', '\x3', '\x2', '\x2', '\x2', 
		'\x43', '\x14C', '\x3', '\x2', '\x2', '\x2', '\x45', '\x14E', '\x3', '\x2', 
		'\x2', '\x2', 'G', '\x156', '\x3', '\x2', '\x2', '\x2', 'I', '\x15F', 
		'\x3', '\x2', '\x2', '\x2', 'K', '\x163', '\x3', '\x2', '\x2', '\x2', 
		'M', '\x165', '\x3', '\x2', '\x2', '\x2', 'O', '\x167', '\x3', '\x2', 
		'\x2', '\x2', 'Q', '\x169', '\x3', '\x2', '\x2', '\x2', 'S', '\x16B', 
		'\x3', '\x2', '\x2', '\x2', 'U', '\x16D', '\x3', '\x2', '\x2', '\x2', 
		'W', '\x16F', '\x3', '\x2', '\x2', '\x2', 'Y', '\x171', '\x3', '\x2', 
		'\x2', '\x2', '[', '\x173', '\x3', '\x2', '\x2', '\x2', ']', '\x175', 
		'\x3', '\x2', '\x2', '\x2', '_', '\x177', '\x3', '\x2', '\x2', '\x2', 
		'\x61', '\x179', '\x3', '\x2', '\x2', '\x2', '\x63', '\x17B', '\x3', '\x2', 
		'\x2', '\x2', '\x65', '\x17D', '\x3', '\x2', '\x2', '\x2', 'g', '\x17F', 
		'\x3', '\x2', '\x2', '\x2', 'i', '\x181', '\x3', '\x2', '\x2', '\x2', 
		'k', '\x183', '\x3', '\x2', '\x2', '\x2', 'm', '\x185', '\x3', '\x2', 
		'\x2', '\x2', 'o', '\x187', '\x3', '\x2', '\x2', '\x2', 'q', '\x189', 
		'\x3', '\x2', '\x2', '\x2', 's', '\x18B', '\x3', '\x2', '\x2', '\x2', 
		'u', '\x18D', '\x3', '\x2', '\x2', '\x2', 'w', '\x18F', '\x3', '\x2', 
		'\x2', '\x2', 'y', '\x191', '\x3', '\x2', '\x2', '\x2', '{', '\x193', 
		'\x3', '\x2', '\x2', '\x2', '}', '\x195', '\x3', '\x2', '\x2', '\x2', 
		'\x7F', '\x197', '\x3', '\x2', '\x2', '\x2', '\x81', '\x83', '\t', '\x2', 
		'\x2', '\x2', '\x82', '\x81', '\x3', '\x2', '\x2', '\x2', '\x83', '\x84', 
		'\x3', '\x2', '\x2', '\x2', '\x84', '\x82', '\x3', '\x2', '\x2', '\x2', 
		'\x84', '\x85', '\x3', '\x2', '\x2', '\x2', '\x85', '\x86', '\x3', '\x2', 
		'\x2', '\x2', '\x86', '\x87', '\b', '\x2', '\x2', '\x2', '\x87', '\x4', 
		'\x3', '\x2', '\x2', '\x2', '\x88', '\x89', '\a', '\x31', '\x2', '\x2', 
		'\x89', '\x8A', '\a', ',', '\x2', '\x2', '\x8A', '\x8B', '\a', '#', '\x2', 
		'\x2', '\x8B', '\x8D', '\x3', '\x2', '\x2', '\x2', '\x8C', '\x8E', '\v', 
		'\x2', '\x2', '\x2', '\x8D', '\x8C', '\x3', '\x2', '\x2', '\x2', '\x8E', 
		'\x8F', '\x3', '\x2', '\x2', '\x2', '\x8F', '\x90', '\x3', '\x2', '\x2', 
		'\x2', '\x8F', '\x8D', '\x3', '\x2', '\x2', '\x2', '\x90', '\x91', '\x3', 
		'\x2', '\x2', '\x2', '\x91', '\x92', '\a', ',', '\x2', '\x2', '\x92', 
		'\x93', '\a', '\x31', '\x2', '\x2', '\x93', '\x94', '\x3', '\x2', '\x2', 
		'\x2', '\x94', '\x95', '\b', '\x3', '\x3', '\x2', '\x95', '\x6', '\x3', 
		'\x2', '\x2', '\x2', '\x96', '\x97', '\a', '\x31', '\x2', '\x2', '\x97', 
		'\x98', '\a', ',', '\x2', '\x2', '\x98', '\x9C', '\x3', '\x2', '\x2', 
		'\x2', '\x99', '\x9B', '\v', '\x2', '\x2', '\x2', '\x9A', '\x99', '\x3', 
		'\x2', '\x2', '\x2', '\x9B', '\x9E', '\x3', '\x2', '\x2', '\x2', '\x9C', 
		'\x9D', '\x3', '\x2', '\x2', '\x2', '\x9C', '\x9A', '\x3', '\x2', '\x2', 
		'\x2', '\x9D', '\x9F', '\x3', '\x2', '\x2', '\x2', '\x9E', '\x9C', '\x3', 
		'\x2', '\x2', '\x2', '\x9F', '\xA0', '\a', ',', '\x2', '\x2', '\xA0', 
		'\xA1', '\a', '\x31', '\x2', '\x2', '\xA1', '\xA2', '\x3', '\x2', '\x2', 
		'\x2', '\xA2', '\xA3', '\b', '\x4', '\x2', '\x2', '\xA3', '\b', '\x3', 
		'\x2', '\x2', '\x2', '\xA4', '\xA5', '\a', '/', '\x2', '\x2', '\xA5', 
		'\xA6', '\a', '/', '\x2', '\x2', '\xA6', '\xA9', '\a', '\"', '\x2', '\x2', 
		'\xA7', '\xA9', '\a', '%', '\x2', '\x2', '\xA8', '\xA4', '\x3', '\x2', 
		'\x2', '\x2', '\xA8', '\xA7', '\x3', '\x2', '\x2', '\x2', '\xA9', '\xAD', 
		'\x3', '\x2', '\x2', '\x2', '\xAA', '\xAC', '\n', '\x3', '\x2', '\x2', 
		'\xAB', '\xAA', '\x3', '\x2', '\x2', '\x2', '\xAC', '\xAF', '\x3', '\x2', 
		'\x2', '\x2', '\xAD', '\xAB', '\x3', '\x2', '\x2', '\x2', '\xAD', '\xAE', 
		'\x3', '\x2', '\x2', '\x2', '\xAE', '\xB5', '\x3', '\x2', '\x2', '\x2', 
		'\xAF', '\xAD', '\x3', '\x2', '\x2', '\x2', '\xB0', '\xB2', '\a', '\xF', 
		'\x2', '\x2', '\xB1', '\xB0', '\x3', '\x2', '\x2', '\x2', '\xB1', '\xB2', 
		'\x3', '\x2', '\x2', '\x2', '\xB2', '\xB3', '\x3', '\x2', '\x2', '\x2', 
		'\xB3', '\xB6', '\a', '\f', '\x2', '\x2', '\xB4', '\xB6', '\a', '\x2', 
		'\x2', '\x3', '\xB5', '\xB1', '\x3', '\x2', '\x2', '\x2', '\xB5', '\xB4', 
		'\x3', '\x2', '\x2', '\x2', '\xB6', '\xC2', '\x3', '\x2', '\x2', '\x2', 
		'\xB7', '\xB8', '\a', '/', '\x2', '\x2', '\xB8', '\xB9', '\a', '/', '\x2', 
		'\x2', '\xB9', '\xBF', '\x3', '\x2', '\x2', '\x2', '\xBA', '\xBC', '\a', 
		'\xF', '\x2', '\x2', '\xBB', '\xBA', '\x3', '\x2', '\x2', '\x2', '\xBB', 
		'\xBC', '\x3', '\x2', '\x2', '\x2', '\xBC', '\xBD', '\x3', '\x2', '\x2', 
		'\x2', '\xBD', '\xC0', '\a', '\f', '\x2', '\x2', '\xBE', '\xC0', '\a', 
		'\x2', '\x2', '\x3', '\xBF', '\xBB', '\x3', '\x2', '\x2', '\x2', '\xBF', 
		'\xBE', '\x3', '\x2', '\x2', '\x2', '\xC0', '\xC2', '\x3', '\x2', '\x2', 
		'\x2', '\xC1', '\xA8', '\x3', '\x2', '\x2', '\x2', '\xC1', '\xB7', '\x3', 
		'\x2', '\x2', '\x2', '\xC2', '\xC3', '\x3', '\x2', '\x2', '\x2', '\xC3', 
		'\xC4', '\b', '\x5', '\x2', '\x2', '\xC4', '\n', '\x3', '\x2', '\x2', 
		'\x2', '\xC5', '\xC6', '\a', '\x30', '\x2', '\x2', '\xC6', '\f', '\x3', 
		'\x2', '\x2', '\x2', '\xC7', '\xC8', '\a', '\x61', '\x2', '\x2', '\xC8', 
		'\xE', '\x3', '\x2', '\x2', '\x2', '\xC9', '\xCA', '\a', ']', '\x2', '\x2', 
		'\xCA', '\x10', '\x3', '\x2', '\x2', '\x2', '\xCB', '\xCC', '\a', '_', 
		'\x2', '\x2', '\xCC', '\x12', '\x3', '\x2', '\x2', '\x2', '\xCD', '\xCE', 
		'\a', '*', '\x2', '\x2', '\xCE', '\x14', '\x3', '\x2', '\x2', '\x2', '\xCF', 
		'\xD0', '\a', '+', '\x2', '\x2', '\xD0', '\x16', '\x3', '\x2', '\x2', 
		'\x2', '\xD1', '\xD2', '\a', '/', '\x2', '\x2', '\xD2', '\x18', '\x3', 
		'\x2', '\x2', '\x2', '\xD3', '\xD4', '\a', ',', '\x2', '\x2', '\xD4', 
		'\x1A', '\x3', '\x2', '\x2', '\x2', '\xD5', '\xD6', '\t', '\x4', '\x2', 
		'\x2', '\xD6', '\x1C', '\x3', '\x2', '\x2', '\x2', '\xD7', '\xD8', '\a', 
		'=', '\x2', '\x2', '\xD8', '\x1E', '\x3', '\x2', '\x2', '\x2', '\xD9', 
		'\xDA', '\a', '@', '\x2', '\x2', '\xDA', ' ', '\x3', '\x2', '\x2', '\x2', 
		'\xDB', '\xDC', '\a', ')', '\x2', '\x2', '\xDC', '\"', '\x3', '\x2', '\x2', 
		'\x2', '\xDD', '\xDE', '\a', '$', '\x2', '\x2', '\xDE', '$', '\x3', '\x2', 
		'\x2', '\x2', '\xDF', '\xE0', '\a', '\x62', '\x2', '\x2', '\xE0', '&', 
		'\x3', '\x2', '\x2', '\x2', '\xE1', '\xE2', '\t', '\x5', '\x2', '\x2', 
		'\xE2', '(', '\x3', '\x2', '\x2', '\x2', '\xE3', '\xE4', '\a', '?', '\x2', 
		'\x2', '\xE4', '*', '\x3', '\x2', '\x2', '\x2', '\xE5', '\xE6', '\a', 
		'#', '\x2', '\x2', '\xE6', '\xE7', '\a', '?', '\x2', '\x2', '\xE7', ',', 
		'\x3', '\x2', '\x2', '\x2', '\xE8', '\xE9', '\a', '~', '\x2', '\x2', '\xE9', 
		'\xEC', '\a', '~', '\x2', '\x2', '\xEA', '\xEC', '\a', '~', '\x2', '\x2', 
		'\xEB', '\xE8', '\x3', '\x2', '\x2', '\x2', '\xEB', '\xEA', '\x3', '\x2', 
		'\x2', '\x2', '\xEC', '.', '\x3', '\x2', '\x2', '\x2', '\xED', '\xEE', 
		'\a', '(', '\x2', '\x2', '\xEE', '\xF2', '\a', '(', '\x2', '\x2', '\xEF', 
		'\xF2', '\x5', '\x1B', '\xE', '\x2', '\xF0', '\xF2', '\a', '(', '\x2', 
		'\x2', '\xF1', '\xED', '\x3', '\x2', '\x2', '\x2', '\xF1', '\xEF', '\x3', 
		'\x2', '\x2', '\x2', '\xF1', '\xF0', '\x3', '\x2', '\x2', '\x2', '\xF2', 
		'\x30', '\x3', '\x2', '\x2', '\x2', '\xF3', '\xF5', '\x5', '\x17', '\f', 
		'\x2', '\xF4', '\xF3', '\x3', '\x2', '\x2', '\x2', '\xF4', '\xF5', '\x3', 
		'\x2', '\x2', '\x2', '\xF5', '\xF7', '\x3', '\x2', '\x2', '\x2', '\xF6', 
		'\xF8', '\x5', 'I', '%', '\x2', '\xF7', '\xF6', '\x3', '\x2', '\x2', '\x2', 
		'\xF8', '\xF9', '\x3', '\x2', '\x2', '\x2', '\xF9', '\xF7', '\x3', '\x2', 
		'\x2', '\x2', '\xF9', '\xFA', '\x3', '\x2', '\x2', '\x2', '\xFA', '\x32', 
		'\x3', '\x2', '\x2', '\x2', '\xFB', '\xFD', '\x5', '\x17', '\f', '\x2', 
		'\xFC', '\xFB', '\x3', '\x2', '\x2', '\x2', '\xFC', '\xFD', '\x3', '\x2', 
		'\x2', '\x2', '\xFD', '\xFF', '\x3', '\x2', '\x2', '\x2', '\xFE', '\x100', 
		'\x5', 'I', '%', '\x2', '\xFF', '\xFE', '\x3', '\x2', '\x2', '\x2', '\x100', 
		'\x101', '\x3', '\x2', '\x2', '\x2', '\x101', '\xFF', '\x3', '\x2', '\x2', 
		'\x2', '\x101', '\x102', '\x3', '\x2', '\x2', '\x2', '\x102', '\x103', 
		'\x3', '\x2', '\x2', '\x2', '\x103', '\x105', '\x5', '\v', '\x6', '\x2', 
		'\x104', '\x106', '\x5', 'I', '%', '\x2', '\x105', '\x104', '\x3', '\x2', 
		'\x2', '\x2', '\x106', '\x107', '\x3', '\x2', '\x2', '\x2', '\x107', '\x105', 
		'\x3', '\x2', '\x2', '\x2', '\x107', '\x108', '\x3', '\x2', '\x2', '\x2', 
		'\x108', '\x113', '\x3', '\x2', '\x2', '\x2', '\x109', '\x10B', '\x5', 
		'\x17', '\f', '\x2', '\x10A', '\x109', '\x3', '\x2', '\x2', '\x2', '\x10A', 
		'\x10B', '\x3', '\x2', '\x2', '\x2', '\x10B', '\x10C', '\x3', '\x2', '\x2', 
		'\x2', '\x10C', '\x10E', '\x5', '\v', '\x6', '\x2', '\x10D', '\x10F', 
		'\x5', 'I', '%', '\x2', '\x10E', '\x10D', '\x3', '\x2', '\x2', '\x2', 
		'\x10F', '\x110', '\x3', '\x2', '\x2', '\x2', '\x110', '\x10E', '\x3', 
		'\x2', '\x2', '\x2', '\x110', '\x111', '\x3', '\x2', '\x2', '\x2', '\x111', 
		'\x113', '\x3', '\x2', '\x2', '\x2', '\x112', '\xFC', '\x3', '\x2', '\x2', 
		'\x2', '\x112', '\x10A', '\x3', '\x2', '\x2', '\x2', '\x113', '\x34', 
		'\x3', '\x2', '\x2', '\x2', '\x114', '\x115', '\x5', 'q', '\x39', '\x2', 
		'\x115', '\x116', '\x5', 'U', '+', '\x2', '\x116', '\x117', '\x5', '\x63', 
		'\x32', '\x2', '\x117', '\x118', '\x5', 'U', '+', '\x2', '\x118', '\x119', 
		'\x5', 'Q', ')', '\x2', '\x119', '\x11A', '\x5', 's', ':', '\x2', '\x11A', 
		'\x36', '\x3', '\x2', '\x2', '\x2', '\x11B', '\x11C', '\x5', 'M', '\'', 
		'\x2', '\x11C', '\x11D', '\x5', '\x63', '\x32', '\x2', '\x11D', '\x11E', 
		'\x5', '\x63', '\x32', '\x2', '\x11E', '\x38', '\x3', '\x2', '\x2', '\x2', 
		'\x11F', '\x120', '\x5', 'S', '*', '\x2', '\x120', '\x121', '\x5', ']', 
		'/', '\x2', '\x121', '\x122', '\x5', 'q', '\x39', '\x2', '\x122', '\x123', 
		'\x5', 's', ':', '\x2', '\x123', '\x124', '\x5', ']', '/', '\x2', '\x124', 
		'\x125', '\x5', 'g', '\x34', '\x2', '\x125', '\x126', '\x5', 'Q', ')', 
		'\x2', '\x126', '\x127', '\x5', 's', ':', '\x2', '\x127', ':', '\x3', 
		'\x2', '\x2', '\x2', '\x128', '\x129', '\x5', 'S', '*', '\x2', '\x129', 
		'\x12A', '\x5', 'U', '+', '\x2', '\x12A', '\x12B', '\x5', '\x63', '\x32', 
		'\x2', '\x12B', '\x12C', '\x5', 'U', '+', '\x2', '\x12C', '\x12D', '\x5', 
		's', ':', '\x2', '\x12D', '\x12E', '\x5', 'U', '+', '\x2', '\x12E', '\x12F', 
		'\x5', 'S', '*', '\x2', '\x12F', '<', '\x3', '\x2', '\x2', '\x2', '\x130', 
		'\x131', '\x5', ']', '/', '\x2', '\x131', '\x132', '\x5', 'g', '\x34', 
		'\x2', '\x132', '\x133', '\x5', 'q', '\x39', '\x2', '\x133', '\x134', 
		'\x5', 'U', '+', '\x2', '\x134', '\x135', '\x5', 'o', '\x38', '\x2', '\x135', 
		'\x136', '\x5', 's', ':', '\x2', '\x136', '\x137', '\x5', 'U', '+', '\x2', 
		'\x137', '\x138', '\x5', 'S', '*', '\x2', '\x138', '>', '\x3', '\x2', 
		'\x2', '\x2', '\x139', '\x13A', '\x5', 's', ':', '\x2', '\x13A', '\x13B', 
		'\x5', 'i', '\x35', '\x2', '\x13B', '\x13C', '\x5', 'k', '\x36', '\x2', 
		'\x13C', '@', '\x3', '\x2', '\x2', '\x2', '\x13D', '\x13E', '\x5', 'M', 
		'\'', '\x2', '\x13E', '\x13F', '\x5', 'q', '\x39', '\x2', '\x13F', '\x42', 
		'\x3', '\x2', '\x2', '\x2', '\x140', '\x142', '\x5', 'K', '&', '\x2', 
		'\x141', '\x140', '\x3', '\x2', '\x2', '\x2', '\x142', '\x143', '\x3', 
		'\x2', '\x2', '\x2', '\x143', '\x141', '\x3', '\x2', '\x2', '\x2', '\x143', 
		'\x144', '\x3', '\x2', '\x2', '\x2', '\x144', '\x14D', '\x3', '\x2', '\x2', 
		'\x2', '\x145', '\x148', '\x5', 'K', '&', '\x2', '\x146', '\x149', '\x5', 
		'K', '&', '\x2', '\x147', '\x149', '\x5', 'I', '%', '\x2', '\x148', '\x146', 
		'\x3', '\x2', '\x2', '\x2', '\x148', '\x147', '\x3', '\x2', '\x2', '\x2', 
		'\x149', '\x14A', '\x3', '\x2', '\x2', '\x2', '\x14A', '\x148', '\x3', 
		'\x2', '\x2', '\x2', '\x14A', '\x14B', '\x3', '\x2', '\x2', '\x2', '\x14B', 
		'\x14D', '\x3', '\x2', '\x2', '\x2', '\x14C', '\x141', '\x3', '\x2', '\x2', 
		'\x2', '\x14C', '\x145', '\x3', '\x2', '\x2', '\x2', '\x14D', '\x44', 
		'\x3', '\x2', '\x2', '\x2', '\x14E', '\x150', '\x5', '\xF', '\b', '\x2', 
		'\x14F', '\x151', '\x5', 'K', '&', '\x2', '\x150', '\x14F', '\x3', '\x2', 
		'\x2', '\x2', '\x151', '\x152', '\x3', '\x2', '\x2', '\x2', '\x152', '\x150', 
		'\x3', '\x2', '\x2', '\x2', '\x152', '\x153', '\x3', '\x2', '\x2', '\x2', 
		'\x153', '\x154', '\x3', '\x2', '\x2', '\x2', '\x154', '\x155', '\x5', 
		'\x11', '\t', '\x2', '\x155', '\x46', '\x3', '\x2', '\x2', '\x2', '\x156', 
		'\x15A', '\x5', '!', '\x11', '\x2', '\x157', '\x159', '\n', '\x6', '\x2', 
		'\x2', '\x158', '\x157', '\x3', '\x2', '\x2', '\x2', '\x159', '\x15C', 
		'\x3', '\x2', '\x2', '\x2', '\x15A', '\x158', '\x3', '\x2', '\x2', '\x2', 
		'\x15A', '\x15B', '\x3', '\x2', '\x2', '\x2', '\x15B', '\x15D', '\x3', 
		'\x2', '\x2', '\x2', '\x15C', '\x15A', '\x3', '\x2', '\x2', '\x2', '\x15D', 
		'\x15E', '\x5', '!', '\x11', '\x2', '\x15E', 'H', '\x3', '\x2', '\x2', 
		'\x2', '\x15F', '\x160', '\t', '\a', '\x2', '\x2', '\x160', 'J', '\x3', 
		'\x2', '\x2', '\x2', '\x161', '\x164', '\t', '\b', '\x2', '\x2', '\x162', 
		'\x164', '\x5', '\r', '\a', '\x2', '\x163', '\x161', '\x3', '\x2', '\x2', 
		'\x2', '\x163', '\x162', '\x3', '\x2', '\x2', '\x2', '\x164', 'L', '\x3', 
		'\x2', '\x2', '\x2', '\x165', '\x166', '\t', '\t', '\x2', '\x2', '\x166', 
		'N', '\x3', '\x2', '\x2', '\x2', '\x167', '\x168', '\t', '\n', '\x2', 
		'\x2', '\x168', 'P', '\x3', '\x2', '\x2', '\x2', '\x169', '\x16A', '\t', 
		'\v', '\x2', '\x2', '\x16A', 'R', '\x3', '\x2', '\x2', '\x2', '\x16B', 
		'\x16C', '\t', '\f', '\x2', '\x2', '\x16C', 'T', '\x3', '\x2', '\x2', 
		'\x2', '\x16D', '\x16E', '\t', '\r', '\x2', '\x2', '\x16E', 'V', '\x3', 
		'\x2', '\x2', '\x2', '\x16F', '\x170', '\t', '\xE', '\x2', '\x2', '\x170', 
		'X', '\x3', '\x2', '\x2', '\x2', '\x171', '\x172', '\t', '\xF', '\x2', 
		'\x2', '\x172', 'Z', '\x3', '\x2', '\x2', '\x2', '\x173', '\x174', '\t', 
		'\x10', '\x2', '\x2', '\x174', '\\', '\x3', '\x2', '\x2', '\x2', '\x175', 
		'\x176', '\t', '\x11', '\x2', '\x2', '\x176', '^', '\x3', '\x2', '\x2', 
		'\x2', '\x177', '\x178', '\t', '\x12', '\x2', '\x2', '\x178', '`', '\x3', 
		'\x2', '\x2', '\x2', '\x179', '\x17A', '\t', '\x13', '\x2', '\x2', '\x17A', 
		'\x62', '\x3', '\x2', '\x2', '\x2', '\x17B', '\x17C', '\t', '\x14', '\x2', 
		'\x2', '\x17C', '\x64', '\x3', '\x2', '\x2', '\x2', '\x17D', '\x17E', 
		'\t', '\x15', '\x2', '\x2', '\x17E', '\x66', '\x3', '\x2', '\x2', '\x2', 
		'\x17F', '\x180', '\t', '\x16', '\x2', '\x2', '\x180', 'h', '\x3', '\x2', 
		'\x2', '\x2', '\x181', '\x182', '\t', '\x17', '\x2', '\x2', '\x182', 'j', 
		'\x3', '\x2', '\x2', '\x2', '\x183', '\x184', '\t', '\x18', '\x2', '\x2', 
		'\x184', 'l', '\x3', '\x2', '\x2', '\x2', '\x185', '\x186', '\t', '\x19', 
		'\x2', '\x2', '\x186', 'n', '\x3', '\x2', '\x2', '\x2', '\x187', '\x188', 
		'\t', '\x1A', '\x2', '\x2', '\x188', 'p', '\x3', '\x2', '\x2', '\x2', 
		'\x189', '\x18A', '\t', '\x1B', '\x2', '\x2', '\x18A', 'r', '\x3', '\x2', 
		'\x2', '\x2', '\x18B', '\x18C', '\t', '\x1C', '\x2', '\x2', '\x18C', 't', 
		'\x3', '\x2', '\x2', '\x2', '\x18D', '\x18E', '\t', '\x1D', '\x2', '\x2', 
		'\x18E', 'v', '\x3', '\x2', '\x2', '\x2', '\x18F', '\x190', '\t', '\x1E', 
		'\x2', '\x2', '\x190', 'x', '\x3', '\x2', '\x2', '\x2', '\x191', '\x192', 
		'\t', '\x1F', '\x2', '\x2', '\x192', 'z', '\x3', '\x2', '\x2', '\x2', 
		'\x193', '\x194', '\t', ' ', '\x2', '\x2', '\x194', '|', '\x3', '\x2', 
		'\x2', '\x2', '\x195', '\x196', '\t', '!', '\x2', '\x2', '\x196', '~', 
		'\x3', '\x2', '\x2', '\x2', '\x197', '\x198', '\t', '\"', '\x2', '\x2', 
		'\x198', '\x80', '\x3', '\x2', '\x2', '\x2', '\x1E', '\x2', '\x84', '\x8F', 
		'\x9C', '\xA8', '\xAD', '\xB1', '\xB5', '\xBB', '\xBF', '\xC1', '\xEB', 
		'\xF1', '\xF4', '\xF9', '\xFC', '\x101', '\x107', '\x10A', '\x110', '\x112', 
		'\x143', '\x148', '\x14A', '\x14C', '\x152', '\x15A', '\x163', '\x4', 
		'\x2', '\x3', '\x2', '\x2', '\x4', '\x2',
	};

	public static readonly ATN _ATN =
		new ATNDeserializer().Deserialize(_serializedATN);


}
} // namespace T1.SqlDom.Tsql
