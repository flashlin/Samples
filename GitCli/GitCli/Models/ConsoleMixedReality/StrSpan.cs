namespace GitCli.Models.ConsoleMixedReality;

public struct StrSpan
{
	public static StrSpan Empty = new StrSpan
	{
		Index = 0,
		Length = 0
	};

	public int Index { get; init; }
	public bool IsEmpty => (Length <= 0);
	public int Length { get; init; }

	public int Right => Index + Length - 1;
	
	public bool Contain(int pos)
	{
		return (pos >= Index && pos < Index + Length);
	}

	public bool Contain(StrSpan span)
	{
		return (span.Index >= Index && span.Right <= Right);
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

	public IEnumerable<StrSpan> NonIntersect(StrSpan b)
	{
		var intersectSpan = Intersect(b);
		if (intersectSpan.IsEmpty)
		{
			yield break;
		}

		if (intersectSpan == this)
		{
			var leftSpan = GetLeftSpan(b);
			if (!leftSpan.IsEmpty)
			{
				yield return leftSpan;
			}

			var rightSpan = GetRightSpan(b);
			if (!rightSpan.IsEmpty)
			{
				yield return rightSpan;
			}
			yield break;
		}

		var leftSpan1 = GetLeftSpan(b);
		if (!leftSpan1.IsEmpty && !Contain(leftSpan1))
		{
			yield return leftSpan1;
		}
		var rightSpan1 = GetRightSpan(b);
		if (!rightSpan1.IsEmpty && !Contain(rightSpan1))
		{
			yield return rightSpan1;
		}
	}

	private StrSpan GetRightSpan(StrSpan b)
	{
		var right = Math.Max(this.Right + 1, b.Index);
		var rightRight = Math.Max(this.Right, b.Right);
		var rightSpan = new StrSpan
		{
			Index = right,
			Length = rightRight- right + 1,
		};
		return rightSpan;
	}

	private StrSpan GetLeftSpan(StrSpan b)
	{
		var left = Math.Min(this.Index, b.Index);
		var leftRight = Math.Max(this.Index, b.Index);
		var leftSpan = new StrSpan
		{
			Index = left,
			Length = leftRight - left
		};
		return leftSpan;
	}

	public override string ToString()
	{
		return $"{nameof(StrSpan)}{{{Index},{Length}}}";
	}

	public static bool operator ==(StrSpan lhs, StrSpan rhs) => lhs.Index == rhs.Index && lhs.Length == rhs.Length;
	public static bool operator !=(StrSpan lhs, StrSpan rhs) => !(lhs == rhs);
}