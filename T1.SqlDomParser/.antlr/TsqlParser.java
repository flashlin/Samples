// Generated from d:\VDisk\Github\Samples\T1.SqlDomParser\TsqlParser.g4 by ANTLR 4.9.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class TsqlParser extends Parser {
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
		RULE_select_statement = 0, RULE_top_clause = 1, RULE_top_count = 2, RULE_column_elem = 3, 
		RULE_full_column_name = 4, RULE_select_list = 5, RULE_id_ = 6, RULE_as_column_alias = 7, 
		RULE_column_alias = 8;
	private static String[] makeRuleNames() {
		return new String[] {
			"select_statement", "top_clause", "top_count", "column_elem", "full_column_name", 
			"select_list", "id_", "as_column_alias", "column_alias"
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

	@Override
	public String getGrammarFileName() { return "TsqlParser.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public TsqlParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class Select_statementContext extends ParserRuleContext {
		public Token allOrDistinct;
		public Top_clauseContext top;
		public Select_listContext columns;
		public TerminalNode SELECT() { return getToken(TsqlParser.SELECT, 0); }
		public Select_listContext select_list() {
			return getRuleContext(Select_listContext.class,0);
		}
		public Top_clauseContext top_clause() {
			return getRuleContext(Top_clauseContext.class,0);
		}
		public TerminalNode ALL() { return getToken(TsqlParser.ALL, 0); }
		public TerminalNode DISTINCT() { return getToken(TsqlParser.DISTINCT, 0); }
		public Select_statementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_select_statement; }
	}

	public final Select_statementContext select_statement() throws RecognitionException {
		Select_statementContext _localctx = new Select_statementContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_select_statement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(18);
			match(SELECT);
			setState(20);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ALL || _la==DISTINCT) {
				{
				setState(19);
				((Select_statementContext)_localctx).allOrDistinct = _input.LT(1);
				_la = _input.LA(1);
				if ( !(_la==ALL || _la==DISTINCT) ) {
					((Select_statementContext)_localctx).allOrDistinct = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				}
			}

			setState(23);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==TOP) {
				{
				setState(22);
				((Select_statementContext)_localctx).top = top_clause();
				}
			}

			setState(25);
			((Select_statementContext)_localctx).columns = select_list();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Top_clauseContext extends ParserRuleContext {
		public TerminalNode TOP() { return getToken(TsqlParser.TOP, 0); }
		public Top_countContext top_count() {
			return getRuleContext(Top_countContext.class,0);
		}
		public Top_clauseContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_top_clause; }
	}

	public final Top_clauseContext top_clause() throws RecognitionException {
		Top_clauseContext _localctx = new Top_clauseContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_top_clause);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(27);
			match(TOP);
			{
			setState(28);
			top_count();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Top_countContext extends ParserRuleContext {
		public Token count_constant;
		public TerminalNode DECIMAL() { return getToken(TsqlParser.DECIMAL, 0); }
		public Top_countContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_top_count; }
	}

	public final Top_countContext top_count() throws RecognitionException {
		Top_countContext _localctx = new Top_countContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_top_count);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(30);
			((Top_countContext)_localctx).count_constant = match(DECIMAL);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Column_elemContext extends ParserRuleContext {
		public Full_column_nameContext full_column_name() {
			return getRuleContext(Full_column_nameContext.class,0);
		}
		public As_column_aliasContext as_column_alias() {
			return getRuleContext(As_column_aliasContext.class,0);
		}
		public Column_elemContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_column_elem; }
	}

	public final Column_elemContext column_elem() throws RecognitionException {
		Column_elemContext _localctx = new Column_elemContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_column_elem);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(32);
			full_column_name();
			}
			setState(34);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << AS) | (1L << ID_LETTER) | (1L << SQUARE_BRACKET_ID) | (1L << STRING))) != 0)) {
				{
				setState(33);
				as_column_alias();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Full_column_nameContext extends ParserRuleContext {
		public Id_Context column_name;
		public Id_Context server;
		public Id_Context schema;
		public Id_Context tablename;
		public List<TerminalNode> DOT() { return getTokens(TsqlParser.DOT); }
		public TerminalNode DOT(int i) {
			return getToken(TsqlParser.DOT, i);
		}
		public TerminalNode DELETED() { return getToken(TsqlParser.DELETED, 0); }
		public TerminalNode INSERTED() { return getToken(TsqlParser.INSERTED, 0); }
		public List<Id_Context> id_() {
			return getRuleContexts(Id_Context.class);
		}
		public Id_Context id_(int i) {
			return getRuleContext(Id_Context.class,i);
		}
		public Full_column_nameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_full_column_name; }
	}

	public final Full_column_nameContext full_column_name() throws RecognitionException {
		Full_column_nameContext _localctx = new Full_column_nameContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_full_column_name);
		int _la;
		try {
			setState(67);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,9,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(36);
				_la = _input.LA(1);
				if ( !(_la==DELETED || _la==INSERTED) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(37);
				match(DOT);
				setState(38);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(40);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(39);
					((Full_column_nameContext)_localctx).server = id_();
					}
				}

				setState(42);
				match(DOT);
				setState(44);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(43);
					((Full_column_nameContext)_localctx).schema = id_();
					}
				}

				setState(46);
				match(DOT);
				setState(48);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(47);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(50);
				match(DOT);
				setState(51);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(53);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(52);
					((Full_column_nameContext)_localctx).schema = id_();
					}
				}

				setState(55);
				match(DOT);
				setState(57);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(56);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(59);
				match(DOT);
				setState(60);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(62);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(61);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(64);
				match(DOT);
				setState(65);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(66);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Select_listContext extends ParserRuleContext {
		public Column_elemContext column_elem;
		public List<Column_elemContext> selectElement = new ArrayList<Column_elemContext>();
		public List<Column_elemContext> column_elem() {
			return getRuleContexts(Column_elemContext.class);
		}
		public Column_elemContext column_elem(int i) {
			return getRuleContext(Column_elemContext.class,i);
		}
		public List<TerminalNode> COMMA() { return getTokens(TsqlParser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(TsqlParser.COMMA, i);
		}
		public Select_listContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_select_list; }
	}

	public final Select_listContext select_list() throws RecognitionException {
		Select_listContext _localctx = new Select_listContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_select_list);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(69);
			((Select_listContext)_localctx).column_elem = column_elem();
			((Select_listContext)_localctx).selectElement.add(((Select_listContext)_localctx).column_elem);
			setState(74);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==COMMA) {
				{
				{
				setState(70);
				match(COMMA);
				setState(71);
				column_elem();
				}
				}
				setState(76);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Id_Context extends ParserRuleContext {
		public TerminalNode ID_LETTER() { return getToken(TsqlParser.ID_LETTER, 0); }
		public TerminalNode SQUARE_BRACKET_ID() { return getToken(TsqlParser.SQUARE_BRACKET_ID, 0); }
		public Id_Context(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_id_; }
	}

	public final Id_Context id_() throws RecognitionException {
		Id_Context _localctx = new Id_Context(_ctx, getState());
		enterRule(_localctx, 12, RULE_id_);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(77);
			_la = _input.LA(1);
			if ( !(_la==ID_LETTER || _la==SQUARE_BRACKET_ID) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class As_column_aliasContext extends ParserRuleContext {
		public Column_aliasContext column_alias() {
			return getRuleContext(Column_aliasContext.class,0);
		}
		public TerminalNode AS() { return getToken(TsqlParser.AS, 0); }
		public As_column_aliasContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_as_column_alias; }
	}

	public final As_column_aliasContext as_column_alias() throws RecognitionException {
		As_column_aliasContext _localctx = new As_column_aliasContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_as_column_alias);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(80);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==AS) {
				{
				setState(79);
				match(AS);
				}
			}

			setState(82);
			column_alias();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Column_aliasContext extends ParserRuleContext {
		public Id_Context id_() {
			return getRuleContext(Id_Context.class,0);
		}
		public TerminalNode STRING() { return getToken(TsqlParser.STRING, 0); }
		public Column_aliasContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_column_alias; }
	}

	public final Column_aliasContext column_alias() throws RecognitionException {
		Column_aliasContext _localctx = new Column_aliasContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_column_alias);
		try {
			setState(86);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case ID_LETTER:
			case SQUARE_BRACKET_ID:
				enterOuterAlt(_localctx, 1);
				{
				setState(84);
				id_();
				}
				break;
			case STRING:
				enterOuterAlt(_localctx, 2);
				{
				setState(85);
				match(STRING);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3%[\4\2\t\2\4\3\t\3"+
		"\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\3\2\3\2\5\2\27"+
		"\n\2\3\2\5\2\32\n\2\3\2\3\2\3\3\3\3\3\3\3\4\3\4\3\5\3\5\5\5%\n\5\3\6\3"+
		"\6\3\6\3\6\5\6+\n\6\3\6\3\6\5\6/\n\6\3\6\3\6\5\6\63\n\6\3\6\3\6\3\6\5"+
		"\68\n\6\3\6\3\6\5\6<\n\6\3\6\3\6\3\6\5\6A\n\6\3\6\3\6\3\6\5\6F\n\6\3\7"+
		"\3\7\3\7\7\7K\n\7\f\7\16\7N\13\7\3\b\3\b\3\t\5\tS\n\t\3\t\3\t\3\n\3\n"+
		"\5\nY\n\n\3\n\2\2\13\2\4\6\b\n\f\16\20\22\2\5\3\2\35\36\3\2\37 \3\2#$"+
		"\2a\2\24\3\2\2\2\4\35\3\2\2\2\6 \3\2\2\2\b\"\3\2\2\2\nE\3\2\2\2\fG\3\2"+
		"\2\2\16O\3\2\2\2\20R\3\2\2\2\22X\3\2\2\2\24\26\7\34\2\2\25\27\t\2\2\2"+
		"\26\25\3\2\2\2\26\27\3\2\2\2\27\31\3\2\2\2\30\32\5\4\3\2\31\30\3\2\2\2"+
		"\31\32\3\2\2\2\32\33\3\2\2\2\33\34\5\f\7\2\34\3\3\2\2\2\35\36\7!\2\2\36"+
		"\37\5\6\4\2\37\5\3\2\2\2 !\7\33\2\2!\7\3\2\2\2\"$\5\n\6\2#%\5\20\t\2$"+
		"#\3\2\2\2$%\3\2\2\2%\t\3\2\2\2&\'\t\3\2\2\'(\7\7\2\2(F\5\16\b\2)+\5\16"+
		"\b\2*)\3\2\2\2*+\3\2\2\2+,\3\2\2\2,.\7\7\2\2-/\5\16\b\2.-\3\2\2\2./\3"+
		"\2\2\2/\60\3\2\2\2\60\62\7\7\2\2\61\63\5\16\b\2\62\61\3\2\2\2\62\63\3"+
		"\2\2\2\63\64\3\2\2\2\64\65\7\7\2\2\65F\5\16\b\2\668\5\16\b\2\67\66\3\2"+
		"\2\2\678\3\2\2\289\3\2\2\29;\7\7\2\2:<\5\16\b\2;:\3\2\2\2;<\3\2\2\2<="+
		"\3\2\2\2=>\7\7\2\2>F\5\16\b\2?A\5\16\b\2@?\3\2\2\2@A\3\2\2\2AB\3\2\2\2"+
		"BC\7\7\2\2CF\5\16\b\2DF\5\16\b\2E&\3\2\2\2E*\3\2\2\2E\67\3\2\2\2E@\3\2"+
		"\2\2ED\3\2\2\2F\13\3\2\2\2GL\5\b\5\2HI\7\17\2\2IK\5\b\5\2JH\3\2\2\2KN"+
		"\3\2\2\2LJ\3\2\2\2LM\3\2\2\2M\r\3\2\2\2NL\3\2\2\2OP\t\4\2\2P\17\3\2\2"+
		"\2QS\7\"\2\2RQ\3\2\2\2RS\3\2\2\2ST\3\2\2\2TU\5\22\n\2U\21\3\2\2\2VY\5"+
		"\16\b\2WY\7%\2\2XV\3\2\2\2XW\3\2\2\2Y\23\3\2\2\2\17\26\31$*.\62\67;@E"+
		"LRX";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}