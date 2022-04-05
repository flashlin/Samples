using System.Collections.Immutable;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class CallSqlDom : SqlDom
	{
		public SqlDom Function { get; }
		public ImmutableArray<SqlDom> Args { get; }

		public CallSqlDom(SqlDom function, ImmutableArray<SqlDom> args) =>
			 (Function, Args) = (function, args);
	}

	//public class SqlValue : SqlDom
	//{
	//	public string ValueType { get; set; }
	//}

	//public class UnarySqlDom : SqlDom
	//{
	//	public string Oper { get; set; }
	//     public SqlDom Right { get; set; }
	//}

	//public interface IInfixParseLet
	//{
	//	SqlDom Handle(SqlDom left, ReadOnlySpan<char> token, IParser parser);
	//	int Precedence { get; }
	//}

	//public interface IPrefixParseLet
	//{
	//	SqlDom Handle(ReadOnlySpan<char> token, IParser parser);
	//}


	//public class SqlParser : IParser
	//{
	//	Dictionary<string, IPrefixParseLet> _prefixParselets = new Dictionary<string, IPrefixParseLet>();

	//	void CreatePrefixOperator(string prefix, int precedence)
	//	{
	//		return new UnarySqlDom
	//		{

	//		};
	//	}

	//	private readonly IScanner _scanner;

	//	public SqlParser(IScanner scanner)
	//	{
	//		this._scanner = scanner;
	//	}

	//	public IScanner Scanner
	//	{
	//		get { return _scanner; }
	//	}

	//	public SqlDom ParseExp(int ctxPrecedence)
	//	{
	//		var prefixToken = _scanner.Consume();
	//		if (prefixToken.IsEmpty)
	//		{
	//			throw new Exception($"expect token but found none");
	//		}

	//		PrefixParse(prefixToken, parser);
	//	}

	//	public SqlDom ParseProgram()
	//	{
	//		return ParseExp(0);
	//	}
	//}
}
