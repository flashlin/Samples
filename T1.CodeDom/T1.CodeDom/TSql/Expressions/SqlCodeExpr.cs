using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.IO;
using System.Text;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public abstract class SqlCodeExpr : IExpression
	{
		public List<CommentSqlCodeExpr> Comments { get; set; } = new List<CommentSqlCodeExpr>();

		public abstract void WriteToStream(IndentStream stream);

		public override string ToString()
		{
			var ms = new MemoryStream();
			var sb = new IndentStream(ms, Encoding.UTF8);
			Comments.WriteToStream(sb);
			WriteToStream(sb);
			sb.Flush();
			ms.Position = 0;
			var sr = new StreamReader(ms);
			return sr.ReadToEnd();
		}
	}
}
