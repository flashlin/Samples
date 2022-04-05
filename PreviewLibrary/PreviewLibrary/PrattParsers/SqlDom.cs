namespace PreviewLibrary.PrattParsers
{
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
}
