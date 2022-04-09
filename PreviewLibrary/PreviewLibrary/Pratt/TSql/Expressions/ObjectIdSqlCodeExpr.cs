using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ObjectIdSqlCodeExpr : SqlCodeExpr
	{
		public string DatabaseName { get; set; }
		public string SchemaName { get; set; }
		public string ObjectName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			if (!string.IsNullOrEmpty(DatabaseName))
			{
				stream.Write(DatabaseName);
				stream.Write(".");
			}

			if (!string.IsNullOrEmpty(SchemaName))
			{
				stream.Write(SchemaName);
				stream.Write(".");
			}

			stream.Write(ObjectName);
		}
	}
}