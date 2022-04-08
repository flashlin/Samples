using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class SizeSqlDom : SqlDom
	{
		public int Size { get; set; }
		public int? Scale { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("(");
			stream.Write(Size);
			if(Scale != null)
			{
				stream.Write($",{Scale}");
			}
			stream.Write(")");
		}
	}
}