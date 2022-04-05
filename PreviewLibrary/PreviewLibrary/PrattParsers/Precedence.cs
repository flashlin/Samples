namespace PreviewLibrary.PrattParsers
{
	static class Precedence
	{
		public const int Assignment = 1;
		public const int Conditional = 2;
		public const int Sum = 3;
		public const int Product = 4;
		public const int Exponent = 5;
		public const int Prefix = 6;
		public const int Postfix = 7;
		public const int Call = 8;
		public const int As = 9;
	}
}
