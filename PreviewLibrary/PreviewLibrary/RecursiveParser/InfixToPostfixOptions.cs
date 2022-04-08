using System;

namespace PreviewLibrary.RecursiveParser
{
	public class InfixToPostfixOptions<T>
	{
		public string[] Operators { get; set; }
		public Func<string> GetCurrentTokenText;
		public Func<bool> NextToken;
		public Func<T> ReadToken;
		public Func<T, string, T, T> CreateBinaryOperator;
	}
}
