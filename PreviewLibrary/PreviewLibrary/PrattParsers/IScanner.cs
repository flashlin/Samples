using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;


//using PrefixParselet = System.Func<
//	System.ReadOnlySpan<char>,
//	PreviewLibrary.PrattParsers.IParser,
//	PreviewLibrary.PrattParsers.SqlDom>;

namespace PreviewLibrary.PrattParsers
{
	public interface IScanner
	{
		TextSpan Peek();
		TextSpan Consume(string expect = null);
		string GetSpanString(TextSpan span);
	}

	public abstract class SqlDom
	{
		public int Offset { get; set; }
		public string Token { get; set; }
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

	//public interface IParser
	//{
	//	SqlDom ParseProgram();
	//	SqlDom ParseExp(int ctxPrecedence);
	//	IScanner Scanner { get; }
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





	//public class ValuePrefixParseLet : IPrefixParseLet
	//{
	//	public SqlDom Handle(ReadOnlySpan<char> token, IParser parser)
	//	{
	//		var tokenStr = token.ToString();
	//		return new SqlValue
	//		{
	//			Offset = parser.Scanner.GetOffset() - tokenStr.Length,
	//			Token = tokenStr,
	//		};
	//	}
	//}

	//public static class Pa
	//{
	//	public static readonly PrefixParselet Group =
	//		 (token, parser) =>
	//		 {
	//			 var expression = parser.ParseExp(0);
	//			 parser.Scanner.Consume(")");
	//			 return expression;
	//		 };
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
