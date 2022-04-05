using System.IO;
using System.Text;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public abstract class SqlDom
	{
		public int Offset { get; set; }
		public string Token { get; set; }
		public abstract void WriteToStream(IndentStream stream);

		public override string ToString()
		{
			var ms = new MemoryStream();
			var sb = new IndentStream(ms, Encoding.UTF8);
			WriteToStream(sb);
			sb.Flush();
			ms.Position = 0;
			var sr = new StreamReader(ms);
			return sr.ReadToEnd();
		}
	}
}
