// Generated from d:\VDisk\Github\Samples\T1.SqlDomParser\TsqlLexer.g4 by ANTLR 4.9.2
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class TsqlLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.9.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		SPACE=1, SPEC_ESSQL_COMMENT=2, COMMENT_INPUT=3, LINE_COMMENT=4, DOT=5, 
		UNDERLINE=6, LBRACKET=7, RBRACKET=8, LPAREN=9, RPAREN=10, MINUS=11, STAR=12, 
		COMMA=13, SEMI=14, GT=15, SINGLE_QUOTE=16, DOUBLE_QUOTE=17, REVERSE_QUOTE=18, 
		COLON=19, EQ=20, NE=21, BOOLOR=22, BOOLAND=23, INT=24, DECIMAL=25, SELECT=26, 
		ALL=27, DISTINCT=28, DELETED=29, INSERTED=30, TOP=31, AS=32, ID_LETTER=33, 
		SQUARE_BRACKET_ID=34, STRING=35;
	public static final int
		ESQLCOMMENT=2;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN", "ESQLCOMMENT"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"SPACE", "SPEC_ESSQL_COMMENT", "COMMENT_INPUT", "LINE_COMMENT", "DOT", 
			"UNDERLINE", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", "MINUS", "STAR", 
			"COMMA", "SEMI", "GT", "SINGLE_QUOTE", "DOUBLE_QUOTE", "REVERSE_QUOTE", 
			"COLON", "EQ", "NE", "BOOLOR", "BOOLAND", "INT", "DECIMAL", "SELECT", 
			"ALL", "DISTINCT", "DELETED", "INSERTED", "TOP", "AS", "ID_LETTER", "SQUARE_BRACKET_ID", 
			"STRING", "DEC_DIGIT", "LETTER", "A", "B", "C", "D", "E", "F", "G", "H", 
			"I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", 
			"W", "X", "Y", "Z"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, null, null, null, null, "'.'", "'_'", "'['", "']'", "'('", "')'", 
			"'-'", "'*'", null, "';'", "'>'", "'''", "'\"'", "'`'", null, "'='", 
			"'!='"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "SPACE", "SPEC_ESSQL_COMMENT", "COMMENT_INPUT", "LINE_COMMENT", 
			"DOT", "UNDERLINE", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", "MINUS", 
			"STAR", "COMMA", "SEMI", "GT", "SINGLE_QUOTE", "DOUBLE_QUOTE", "REVERSE_QUOTE", 
			"COLON", "EQ", "NE", "BOOLOR", "BOOLAND", "INT", "DECIMAL", "SELECT", 
			"ALL", "DISTINCT", "DELETED", "INSERTED", "TOP", "AS", "ID_LETTER", "SQUARE_BRACKET_ID", 
			"STRING"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public TsqlLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "TsqlLexer.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2%\u0199\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4"+
		",\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64\t"+
		"\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\4:\t:\4;\t;\4<\t<\4=\t="+
		"\4>\t>\4?\t?\4@\t@\3\2\6\2\u0083\n\2\r\2\16\2\u0084\3\2\3\2\3\3\3\3\3"+
		"\3\3\3\3\3\6\3\u008e\n\3\r\3\16\3\u008f\3\3\3\3\3\3\3\3\3\3\3\4\3\4\3"+
		"\4\3\4\7\4\u009b\n\4\f\4\16\4\u009e\13\4\3\4\3\4\3\4\3\4\3\4\3\5\3\5\3"+
		"\5\3\5\5\5\u00a9\n\5\3\5\7\5\u00ac\n\5\f\5\16\5\u00af\13\5\3\5\5\5\u00b2"+
		"\n\5\3\5\3\5\5\5\u00b6\n\5\3\5\3\5\3\5\3\5\5\5\u00bc\n\5\3\5\3\5\5\5\u00c0"+
		"\n\5\5\5\u00c2\n\5\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\n\3\n\3\13"+
		"\3\13\3\f\3\f\3\r\3\r\3\16\3\16\3\17\3\17\3\20\3\20\3\21\3\21\3\22\3\22"+
		"\3\23\3\23\3\24\3\24\3\25\3\25\3\26\3\26\3\26\3\27\3\27\3\27\5\27\u00ec"+
		"\n\27\3\30\3\30\3\30\3\30\5\30\u00f2\n\30\3\31\5\31\u00f5\n\31\3\31\6"+
		"\31\u00f8\n\31\r\31\16\31\u00f9\3\32\5\32\u00fd\n\32\3\32\6\32\u0100\n"+
		"\32\r\32\16\32\u0101\3\32\3\32\6\32\u0106\n\32\r\32\16\32\u0107\3\32\5"+
		"\32\u010b\n\32\3\32\3\32\6\32\u010f\n\32\r\32\16\32\u0110\5\32\u0113\n"+
		"\32\3\33\3\33\3\33\3\33\3\33\3\33\3\33\3\34\3\34\3\34\3\34\3\35\3\35\3"+
		"\35\3\35\3\35\3\35\3\35\3\35\3\35\3\36\3\36\3\36\3\36\3\36\3\36\3\36\3"+
		"\36\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3 \3 \3 \3 \3!\3!\3!"+
		"\3\"\6\"\u0142\n\"\r\"\16\"\u0143\3\"\3\"\3\"\6\"\u0149\n\"\r\"\16\"\u014a"+
		"\5\"\u014d\n\"\3#\3#\6#\u0151\n#\r#\16#\u0152\3#\3#\3$\3$\7$\u0159\n$"+
		"\f$\16$\u015c\13$\3$\3$\3%\3%\3&\3&\5&\u0164\n&\3\'\3\'\3(\3(\3)\3)\3"+
		"*\3*\3+\3+\3,\3,\3-\3-\3.\3.\3/\3/\3\60\3\60\3\61\3\61\3\62\3\62\3\63"+
		"\3\63\3\64\3\64\3\65\3\65\3\66\3\66\3\67\3\67\38\38\39\39\3:\3:\3;\3;"+
		"\3<\3<\3=\3=\3>\3>\3?\3?\3@\3@\4\u008f\u009c\2A\3\3\5\4\7\5\t\6\13\7\r"+
		"\b\17\t\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25"+
		")\26+\27-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I\2K\2M\2"+
		"O\2Q\2S\2U\2W\2Y\2[\2]\2_\2a\2c\2e\2g\2i\2k\2m\2o\2q\2s\2u\2w\2y\2{\2"+
		"}\2\177\2\3\2#\5\2\13\f\17\17\"\"\4\2\f\f\17\17\4\2..\uff0e\uff0e\4\2"+
		"<<\uff1c\uff1c\3\2))\3\2\62;\4\2C\\c|\4\2CCcc\4\2DDdd\4\2EEee\4\2FFff"+
		"\4\2GGgg\4\2HHhh\4\2IIii\4\2JJjj\4\2KKkk\4\2LLll\4\2MMmm\4\2NNnn\4\2O"+
		"Ooo\4\2PPpp\4\2QQqq\4\2RRrr\4\2SSss\4\2TTtt\4\2UUuu\4\2VVvv\4\2WWww\4"+
		"\2XXxx\4\2YYyy\4\2ZZzz\4\2[[{{\4\2\\\\||\2\u0198\2\3\3\2\2\2\2\5\3\2\2"+
		"\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21"+
		"\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2"+
		"\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3"+
		"\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3"+
		"\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3"+
		"\2\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\3\u0082\3\2\2\2\5\u0088"+
		"\3\2\2\2\7\u0096\3\2\2\2\t\u00c1\3\2\2\2\13\u00c5\3\2\2\2\r\u00c7\3\2"+
		"\2\2\17\u00c9\3\2\2\2\21\u00cb\3\2\2\2\23\u00cd\3\2\2\2\25\u00cf\3\2\2"+
		"\2\27\u00d1\3\2\2\2\31\u00d3\3\2\2\2\33\u00d5\3\2\2\2\35\u00d7\3\2\2\2"+
		"\37\u00d9\3\2\2\2!\u00db\3\2\2\2#\u00dd\3\2\2\2%\u00df\3\2\2\2\'\u00e1"+
		"\3\2\2\2)\u00e3\3\2\2\2+\u00e5\3\2\2\2-\u00eb\3\2\2\2/\u00f1\3\2\2\2\61"+
		"\u00f4\3\2\2\2\63\u0112\3\2\2\2\65\u0114\3\2\2\2\67\u011b\3\2\2\29\u011f"+
		"\3\2\2\2;\u0128\3\2\2\2=\u0130\3\2\2\2?\u0139\3\2\2\2A\u013d\3\2\2\2C"+
		"\u014c\3\2\2\2E\u014e\3\2\2\2G\u0156\3\2\2\2I\u015f\3\2\2\2K\u0163\3\2"+
		"\2\2M\u0165\3\2\2\2O\u0167\3\2\2\2Q\u0169\3\2\2\2S\u016b\3\2\2\2U\u016d"+
		"\3\2\2\2W\u016f\3\2\2\2Y\u0171\3\2\2\2[\u0173\3\2\2\2]\u0175\3\2\2\2_"+
		"\u0177\3\2\2\2a\u0179\3\2\2\2c\u017b\3\2\2\2e\u017d\3\2\2\2g\u017f\3\2"+
		"\2\2i\u0181\3\2\2\2k\u0183\3\2\2\2m\u0185\3\2\2\2o\u0187\3\2\2\2q\u0189"+
		"\3\2\2\2s\u018b\3\2\2\2u\u018d\3\2\2\2w\u018f\3\2\2\2y\u0191\3\2\2\2{"+
		"\u0193\3\2\2\2}\u0195\3\2\2\2\177\u0197\3\2\2\2\u0081\u0083\t\2\2\2\u0082"+
		"\u0081\3\2\2\2\u0083\u0084\3\2\2\2\u0084\u0082\3\2\2\2\u0084\u0085\3\2"+
		"\2\2\u0085\u0086\3\2\2\2\u0086\u0087\b\2\2\2\u0087\4\3\2\2\2\u0088\u0089"+
		"\7\61\2\2\u0089\u008a\7,\2\2\u008a\u008b\7#\2\2\u008b\u008d\3\2\2\2\u008c"+
		"\u008e\13\2\2\2\u008d\u008c\3\2\2\2\u008e\u008f\3\2\2\2\u008f\u0090\3"+
		"\2\2\2\u008f\u008d\3\2\2\2\u0090\u0091\3\2\2\2\u0091\u0092\7,\2\2\u0092"+
		"\u0093\7\61\2\2\u0093\u0094\3\2\2\2\u0094\u0095\b\3\3\2\u0095\6\3\2\2"+
		"\2\u0096\u0097\7\61\2\2\u0097\u0098\7,\2\2\u0098\u009c\3\2\2\2\u0099\u009b"+
		"\13\2\2\2\u009a\u0099\3\2\2\2\u009b\u009e\3\2\2\2\u009c\u009d\3\2\2\2"+
		"\u009c\u009a\3\2\2\2\u009d\u009f\3\2\2\2\u009e\u009c\3\2\2\2\u009f\u00a0"+
		"\7,\2\2\u00a0\u00a1\7\61\2\2\u00a1\u00a2\3\2\2\2\u00a2\u00a3\b\4\2\2\u00a3"+
		"\b\3\2\2\2\u00a4\u00a5\7/\2\2\u00a5\u00a6\7/\2\2\u00a6\u00a9\7\"\2\2\u00a7"+
		"\u00a9\7%\2\2\u00a8\u00a4\3\2\2\2\u00a8\u00a7\3\2\2\2\u00a9\u00ad\3\2"+
		"\2\2\u00aa\u00ac\n\3\2\2\u00ab\u00aa\3\2\2\2\u00ac\u00af\3\2\2\2\u00ad"+
		"\u00ab\3\2\2\2\u00ad\u00ae\3\2\2\2\u00ae\u00b5\3\2\2\2\u00af\u00ad\3\2"+
		"\2\2\u00b0\u00b2\7\17\2\2\u00b1\u00b0\3\2\2\2\u00b1\u00b2\3\2\2\2\u00b2"+
		"\u00b3\3\2\2\2\u00b3\u00b6\7\f\2\2\u00b4\u00b6\7\2\2\3\u00b5\u00b1\3\2"+
		"\2\2\u00b5\u00b4\3\2\2\2\u00b6\u00c2\3\2\2\2\u00b7\u00b8\7/\2\2\u00b8"+
		"\u00b9\7/\2\2\u00b9\u00bf\3\2\2\2\u00ba\u00bc\7\17\2\2\u00bb\u00ba\3\2"+
		"\2\2\u00bb\u00bc\3\2\2\2\u00bc\u00bd\3\2\2\2\u00bd\u00c0\7\f\2\2\u00be"+
		"\u00c0\7\2\2\3\u00bf\u00bb\3\2\2\2\u00bf\u00be\3\2\2\2\u00c0\u00c2\3\2"+
		"\2\2\u00c1\u00a8\3\2\2\2\u00c1\u00b7\3\2\2\2\u00c2\u00c3\3\2\2\2\u00c3"+
		"\u00c4\b\5\2\2\u00c4\n\3\2\2\2\u00c5\u00c6\7\60\2\2\u00c6\f\3\2\2\2\u00c7"+
		"\u00c8\7a\2\2\u00c8\16\3\2\2\2\u00c9\u00ca\7]\2\2\u00ca\20\3\2\2\2\u00cb"+
		"\u00cc\7_\2\2\u00cc\22\3\2\2\2\u00cd\u00ce\7*\2\2\u00ce\24\3\2\2\2\u00cf"+
		"\u00d0\7+\2\2\u00d0\26\3\2\2\2\u00d1\u00d2\7/\2\2\u00d2\30\3\2\2\2\u00d3"+
		"\u00d4\7,\2\2\u00d4\32\3\2\2\2\u00d5\u00d6\t\4\2\2\u00d6\34\3\2\2\2\u00d7"+
		"\u00d8\7=\2\2\u00d8\36\3\2\2\2\u00d9\u00da\7@\2\2\u00da \3\2\2\2\u00db"+
		"\u00dc\7)\2\2\u00dc\"\3\2\2\2\u00dd\u00de\7$\2\2\u00de$\3\2\2\2\u00df"+
		"\u00e0\7b\2\2\u00e0&\3\2\2\2\u00e1\u00e2\t\5\2\2\u00e2(\3\2\2\2\u00e3"+
		"\u00e4\7?\2\2\u00e4*\3\2\2\2\u00e5\u00e6\7#\2\2\u00e6\u00e7\7?\2\2\u00e7"+
		",\3\2\2\2\u00e8\u00e9\7~\2\2\u00e9\u00ec\7~\2\2\u00ea\u00ec\7~\2\2\u00eb"+
		"\u00e8\3\2\2\2\u00eb\u00ea\3\2\2\2\u00ec.\3\2\2\2\u00ed\u00ee\7(\2\2\u00ee"+
		"\u00f2\7(\2\2\u00ef\u00f2\5\33\16\2\u00f0\u00f2\7(\2\2\u00f1\u00ed\3\2"+
		"\2\2\u00f1\u00ef\3\2\2\2\u00f1\u00f0\3\2\2\2\u00f2\60\3\2\2\2\u00f3\u00f5"+
		"\5\27\f\2\u00f4\u00f3\3\2\2\2\u00f4\u00f5\3\2\2\2\u00f5\u00f7\3\2\2\2"+
		"\u00f6\u00f8\5I%\2\u00f7\u00f6\3\2\2\2\u00f8\u00f9\3\2\2\2\u00f9\u00f7"+
		"\3\2\2\2\u00f9\u00fa\3\2\2\2\u00fa\62\3\2\2\2\u00fb\u00fd\5\27\f\2\u00fc"+
		"\u00fb\3\2\2\2\u00fc\u00fd\3\2\2\2\u00fd\u00ff\3\2\2\2\u00fe\u0100\5I"+
		"%\2\u00ff\u00fe\3\2\2\2\u0100\u0101\3\2\2\2\u0101\u00ff\3\2\2\2\u0101"+
		"\u0102\3\2\2\2\u0102\u0103\3\2\2\2\u0103\u0105\5\13\6\2\u0104\u0106\5"+
		"I%\2\u0105\u0104\3\2\2\2\u0106\u0107\3\2\2\2\u0107\u0105\3\2\2\2\u0107"+
		"\u0108\3\2\2\2\u0108\u0113\3\2\2\2\u0109\u010b\5\27\f\2\u010a\u0109\3"+
		"\2\2\2\u010a\u010b\3\2\2\2\u010b\u010c\3\2\2\2\u010c\u010e\5\13\6\2\u010d"+
		"\u010f\5I%\2\u010e\u010d\3\2\2\2\u010f\u0110\3\2\2\2\u0110\u010e\3\2\2"+
		"\2\u0110\u0111\3\2\2\2\u0111\u0113\3\2\2\2\u0112\u00fc\3\2\2\2\u0112\u010a"+
		"\3\2\2\2\u0113\64\3\2\2\2\u0114\u0115\5q9\2\u0115\u0116\5U+\2\u0116\u0117"+
		"\5c\62\2\u0117\u0118\5U+\2\u0118\u0119\5Q)\2\u0119\u011a\5s:\2\u011a\66"+
		"\3\2\2\2\u011b\u011c\5M\'\2\u011c\u011d\5c\62\2\u011d\u011e\5c\62\2\u011e"+
		"8\3\2\2\2\u011f\u0120\5S*\2\u0120\u0121\5]/\2\u0121\u0122\5q9\2\u0122"+
		"\u0123\5s:\2\u0123\u0124\5]/\2\u0124\u0125\5g\64\2\u0125\u0126\5Q)\2\u0126"+
		"\u0127\5s:\2\u0127:\3\2\2\2\u0128\u0129\5S*\2\u0129\u012a\5U+\2\u012a"+
		"\u012b\5c\62\2\u012b\u012c\5U+\2\u012c\u012d\5s:\2\u012d\u012e\5U+\2\u012e"+
		"\u012f\5S*\2\u012f<\3\2\2\2\u0130\u0131\5]/\2\u0131\u0132\5g\64\2\u0132"+
		"\u0133\5q9\2\u0133\u0134\5U+\2\u0134\u0135\5o8\2\u0135\u0136\5s:\2\u0136"+
		"\u0137\5U+\2\u0137\u0138\5S*\2\u0138>\3\2\2\2\u0139\u013a\5s:\2\u013a"+
		"\u013b\5i\65\2\u013b\u013c\5k\66\2\u013c@\3\2\2\2\u013d\u013e\5M\'\2\u013e"+
		"\u013f\5q9\2\u013fB\3\2\2\2\u0140\u0142\5K&\2\u0141\u0140\3\2\2\2\u0142"+
		"\u0143\3\2\2\2\u0143\u0141\3\2\2\2\u0143\u0144\3\2\2\2\u0144\u014d\3\2"+
		"\2\2\u0145\u0148\5K&\2\u0146\u0149\5K&\2\u0147\u0149\5I%\2\u0148\u0146"+
		"\3\2\2\2\u0148\u0147\3\2\2\2\u0149\u014a\3\2\2\2\u014a\u0148\3\2\2\2\u014a"+
		"\u014b\3\2\2\2\u014b\u014d\3\2\2\2\u014c\u0141\3\2\2\2\u014c\u0145\3\2"+
		"\2\2\u014dD\3\2\2\2\u014e\u0150\5\17\b\2\u014f\u0151\5K&\2\u0150\u014f"+
		"\3\2\2\2\u0151\u0152\3\2\2\2\u0152\u0150\3\2\2\2\u0152\u0153\3\2\2\2\u0153"+
		"\u0154\3\2\2\2\u0154\u0155\5\21\t\2\u0155F\3\2\2\2\u0156\u015a\5!\21\2"+
		"\u0157\u0159\n\6\2\2\u0158\u0157\3\2\2\2\u0159\u015c\3\2\2\2\u015a\u0158"+
		"\3\2\2\2\u015a\u015b\3\2\2\2\u015b\u015d\3\2\2\2\u015c\u015a\3\2\2\2\u015d"+
		"\u015e\5!\21\2\u015eH\3\2\2\2\u015f\u0160\t\7\2\2\u0160J\3\2\2\2\u0161"+
		"\u0164\t\b\2\2\u0162\u0164\5\r\7\2\u0163\u0161\3\2\2\2\u0163\u0162\3\2"+
		"\2\2\u0164L\3\2\2\2\u0165\u0166\t\t\2\2\u0166N\3\2\2\2\u0167\u0168\t\n"+
		"\2\2\u0168P\3\2\2\2\u0169\u016a\t\13\2\2\u016aR\3\2\2\2\u016b\u016c\t"+
		"\f\2\2\u016cT\3\2\2\2\u016d\u016e\t\r\2\2\u016eV\3\2\2\2\u016f\u0170\t"+
		"\16\2\2\u0170X\3\2\2\2\u0171\u0172\t\17\2\2\u0172Z\3\2\2\2\u0173\u0174"+
		"\t\20\2\2\u0174\\\3\2\2\2\u0175\u0176\t\21\2\2\u0176^\3\2\2\2\u0177\u0178"+
		"\t\22\2\2\u0178`\3\2\2\2\u0179\u017a\t\23\2\2\u017ab\3\2\2\2\u017b\u017c"+
		"\t\24\2\2\u017cd\3\2\2\2\u017d\u017e\t\25\2\2\u017ef\3\2\2\2\u017f\u0180"+
		"\t\26\2\2\u0180h\3\2\2\2\u0181\u0182\t\27\2\2\u0182j\3\2\2\2\u0183\u0184"+
		"\t\30\2\2\u0184l\3\2\2\2\u0185\u0186\t\31\2\2\u0186n\3\2\2\2\u0187\u0188"+
		"\t\32\2\2\u0188p\3\2\2\2\u0189\u018a\t\33\2\2\u018ar\3\2\2\2\u018b\u018c"+
		"\t\34\2\2\u018ct\3\2\2\2\u018d\u018e\t\35\2\2\u018ev\3\2\2\2\u018f\u0190"+
		"\t\36\2\2\u0190x\3\2\2\2\u0191\u0192\t\37\2\2\u0192z\3\2\2\2\u0193\u0194"+
		"\t \2\2\u0194|\3\2\2\2\u0195\u0196\t!\2\2\u0196~\3\2\2\2\u0197\u0198\t"+
		"\"\2\2\u0198\u0080\3\2\2\2\36\2\u0084\u008f\u009c\u00a8\u00ad\u00b1\u00b5"+
		"\u00bb\u00bf\u00c1\u00eb\u00f1\u00f4\u00f9\u00fc\u0101\u0107\u010a\u0110"+
		"\u0112\u0143\u0148\u014a\u014c\u0152\u015a\u0163\4\2\3\2\2\4\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}