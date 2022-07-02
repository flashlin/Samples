namespace GitCli.Models.ConsoleMixedReality;

public struct StrSpan
{
	public static StrSpan Empty = new StrSpan
	{
		Index = 0,
		Length = 0
	};

	public int Index { get; init; }
	public bool IsEmpty => (Index == 0 && Length == 0);
	public int Length { get; init; }

	public int Right => Index + Length - 1;
	public bool Contain(int pos)
	{
		return (pos >= Index && pos < Index + Length);
	}

	public StrSpan Intersect(StrSpan b)
	{
		if (this.IsEmpty || b.IsEmpty) return Empty;
		return new StrSpan
		{
			Index = Math.Max(this.Index, b.Index),
			Length = Math.Min(this.Right, b.Right) - Math.Max(this.Index, b.Index) + 1,
		};
	}
	public override string ToString()
	{
		return $"{nameof(StrSpan)}{{{Index},{Length}}}";
	}
}