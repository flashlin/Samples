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
		RULE_select_statement = 0, RULE_top_clause = 1, RULE_top_count = 2, RULE_constant = 3, 
		RULE_column_elem = 4, RULE_full_column_name = 5, RULE_select_list = 6, 
		RULE_id_ = 7, RULE_as_column_alias = 8, RULE_column_alias = 9;
	private static String[] makeRuleNames() {
		return new String[] {
			"select_statement", "top_clause", "top_count", "constant", "column_elem", 
			"full_column_name", "select_list", "id_", "as_column_alias", "column_alias"
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
			setState(20);
			match(SELECT);
			setState(22);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ALL || _la==DISTINCT) {
				{
				setState(21);
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

			setState(25);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==TOP) {
				{
				setState(24);
				((Select_statementContext)_localctx).top = top_clause();
				}
			}

			setState(27);
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
			setState(29);
			match(TOP);
			{
			setState(30);
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
			setState(32);
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

	public static class ConstantContext extends ParserRuleContext {
		public TerminalNode DECIMAL() { return getToken(TsqlParser.DECIMAL, 0); }
		public TerminalNode INT() { return getToken(TsqlParser.INT, 0); }
		public ConstantContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_constant; }
	}

	public final ConstantContext constant() throws RecognitionException {
		ConstantContext _localctx = new ConstantContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_constant);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(34);
			_la = _input.LA(1);
			if ( !(_la==INT || _la==DECIMAL) ) {
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

	public static class Column_elemContext extends ParserRuleContext {
		public Full_column_nameContext full_column_name() {
			return getRuleContext(Full_column_nameContext.class,0);
		}
		public ConstantContext constant() {
			return getRuleContext(ConstantContext.class,0);
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
		enterRule(_localctx, 8, RULE_column_elem);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(38);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case DOT:
			case DELETED:
			case INSERTED:
			case ID_LETTER:
			case SQUARE_BRACKET_ID:
				{
				setState(36);
				full_column_name();
				}
				break;
			case INT:
			case DECIMAL:
				{
				setState(37);
				constant();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(41);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << AS) | (1L << ID_LETTER) | (1L << SQUARE_BRACKET_ID) | (1L << STRING))) != 0)) {
				{
				setState(40);
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
		enterRule(_localctx, 10, RULE_full_column_name);
		int _la;
		try {
			setState(74);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,10,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(43);
				_la = _input.LA(1);
				if ( !(_la==DELETED || _la==INSERTED) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(44);
				match(DOT);
				setState(45);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(47);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(46);
					((Full_column_nameContext)_localctx).server = id_();
					}
				}

				setState(49);
				match(DOT);
				setState(51);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(50);
					((Full_column_nameContext)_localctx).schema = id_();
					}
				}

				setState(53);
				match(DOT);
				setState(55);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(54);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(57);
				match(DOT);
				setState(58);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(60);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(59);
					((Full_column_nameContext)_localctx).schema = id_();
					}
				}

				setState(62);
				match(DOT);
				setState(64);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(63);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(66);
				match(DOT);
				setState(67);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(69);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==ID_LETTER || _la==SQUARE_BRACKET_ID) {
					{
					setState(68);
					((Full_column_nameContext)_localctx).tablename = id_();
					}
				}

				setState(71);
				match(DOT);
				setState(72);
				((Full_column_nameContext)_localctx).column_name = id_();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(73);
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
		enterRule(_localctx, 12, RULE_select_list);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(76);
			((Select_listContext)_localctx).column_elem = column_elem();
			((Select_listContext)_localctx).selectElement.add(((Select_listContext)_localctx).column_elem);
			setState(81);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==COMMA) {
				{
				{
				setState(77);
				match(COMMA);
				setState(78);
				column_elem();
				}
				}
				setState(83);
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
		enterRule(_localctx, 14, RULE_id_);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(84);
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
		enterRule(_localctx, 16, RULE_as_column_alias);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(87);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==AS) {
				{
				setState(86);
				match(AS);
				}
			}

			setState(89);
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
		enterRule(_localctx, 18, RULE_column_alias);
		try {
			setState(93);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case ID_LETTER:
			case SQUARE_BRACKET_ID:
				enterOuterAlt(_localctx, 1);
				{
				setState(91);
				id_();
				}
				break;
			case STRING:
				enterOuterAlt(_localctx, 2);
				{
				setState(92);
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
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3%b\4\2\t\2\4\3\t\3"+
		"\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\3\2"+
		"\3\2\5\2\31\n\2\3\2\5\2\34\n\2\3\2\3\2\3\3\3\3\3\3\3\4\3\4\3\5\3\5\3\6"+
		"\3\6\5\6)\n\6\3\6\5\6,\n\6\3\7\3\7\3\7\3\7\5\7\62\n\7\3\7\3\7\5\7\66\n"+
		"\7\3\7\3\7\5\7:\n\7\3\7\3\7\3\7\5\7?\n\7\3\7\3\7\5\7C\n\7\3\7\3\7\3\7"+
		"\5\7H\n\7\3\7\3\7\3\7\5\7M\n\7\3\b\3\b\3\b\7\bR\n\b\f\b\16\bU\13\b\3\t"+
		"\3\t\3\n\5\nZ\n\n\3\n\3\n\3\13\3\13\5\13`\n\13\3\13\2\2\f\2\4\6\b\n\f"+
		"\16\20\22\24\2\6\3\2\35\36\3\2\32\33\3\2\37 \3\2#$\2h\2\26\3\2\2\2\4\37"+
		"\3\2\2\2\6\"\3\2\2\2\b$\3\2\2\2\n(\3\2\2\2\fL\3\2\2\2\16N\3\2\2\2\20V"+
		"\3\2\2\2\22Y\3\2\2\2\24_\3\2\2\2\26\30\7\34\2\2\27\31\t\2\2\2\30\27\3"+
		"\2\2\2\30\31\3\2\2\2\31\33\3\2\2\2\32\34\5\4\3\2\33\32\3\2\2\2\33\34\3"+
		"\2\2\2\34\35\3\2\2\2\35\36\5\16\b\2\36\3\3\2\2\2\37 \7!\2\2 !\5\6\4\2"+
		"!\5\3\2\2\2\"#\7\33\2\2#\7\3\2\2\2$%\t\3\2\2%\t\3\2\2\2&)\5\f\7\2\')\5"+
		"\b\5\2(&\3\2\2\2(\'\3\2\2\2)+\3\2\2\2*,\5\22\n\2+*\3\2\2\2+,\3\2\2\2,"+
		"\13\3\2\2\2-.\t\4\2\2./\7\7\2\2/M\5\20\t\2\60\62\5\20\t\2\61\60\3\2\2"+
		"\2\61\62\3\2\2\2\62\63\3\2\2\2\63\65\7\7\2\2\64\66\5\20\t\2\65\64\3\2"+
		"\2\2\65\66\3\2\2\2\66\67\3\2\2\2\679\7\7\2\28:\5\20\t\298\3\2\2\29:\3"+
		"\2\2\2:;\3\2\2\2;<\7\7\2\2<M\5\20\t\2=?\5\20\t\2>=\3\2\2\2>?\3\2\2\2?"+
		"@\3\2\2\2@B\7\7\2\2AC\5\20\t\2BA\3\2\2\2BC\3\2\2\2CD\3\2\2\2DE\7\7\2\2"+
		"EM\5\20\t\2FH\5\20\t\2GF\3\2\2\2GH\3\2\2\2HI\3\2\2\2IJ\7\7\2\2JM\5\20"+
		"\t\2KM\5\20\t\2L-\3\2\2\2L\61\3\2\2\2L>\3\2\2\2LG\3\2\2\2LK\3\2\2\2M\r"+
		"\3\2\2\2NS\5\n\6\2OP\7\17\2\2PR\5\n\6\2QO\3\2\2\2RU\3\2\2\2SQ\3\2\2\2"+
		"ST\3\2\2\2T\17\3\2\2\2US\3\2\2\2VW\t\5\2\2W\21\3\2\2\2XZ\7\"\2\2YX\3\2"+
		"\2\2YZ\3\2\2\2Z[\3\2\2\2[\\\5\24\13\2\\\23\3\2\2\2]`\5\20\t\2^`\7%\2\2"+
		"_]\3\2\2\2_^\3\2\2\2`\25\3\2\2\2\20\30\33(+\61\659>BGLSY_";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}