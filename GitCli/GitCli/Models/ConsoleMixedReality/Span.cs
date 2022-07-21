namespace GitCli.Models.ConsoleMixedReality;

public struct Span
{
	public static Span Empty = new Span
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

	public bool Contain(Span span)
	{
		return (span.Index >= Index && span.Right <= Right);
	}

	public Span Intersect(Span b)
	{
		if (this.IsEmpty || b.IsEmpty) return Empty;
		return new Span
		{
			Index = Math.Max(this.Index, b.Index),
			Length = Math.Min(this.Right, b.Right) - Math.Max(this.Index, b.Index) + 1,
		};
	}

	public IEnumerable<Span> NonIntersect(Span b)
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

	private Span GetRightSpan(Span b)
	{
		var right = Math.Max(this.Right + 1, b.Index);
		var rightRight = Math.Max(this.Right, b.Right);
		var rightSpan = new Span
		{
			Index = right,
			Length = rightRight - right + 1,
		};
		return rightSpan;
	}

	private Span GetLeftSpan(Span b)
	{
		var left = Math.Min(this.Index, b.Index);
		var leftRight = Math.Max(this.Index, b.Index);
		var leftSpan = new Span
		{
			Index = left,
			Length = leftRight - left
		};
		return leftSpan;
	}

	public override string ToString()
	{
		return $"{nameof(Span)}{{{Index},{Length}}}";
	}

	public static bool operator ==(Span lhs, Span rhs) => lhs.Index == rhs.Index && lhs.Length == rhs.Length;
	public static bool operator !=(Span lhs, Span rhs) => !(lhs == rhs);

	public override bool Equals(object? obj)
	{
		if (obj == null)
		{
			return false;
		}

		if (obj is Span b)
		{
			return this == b;
		}

		return false;
	}

	public override int GetHashCode()
	{
		return HashCodeCalculator.GetHashCode(
			 (typeof(int), Index),
			 (typeof(int), Length));
	}

	public Span Move(int add)
	{
		if (Index == 0 && add < 0)
		{
			return this;
		}
		return new Span()
		{
			Index = Index + add,
			Length = Length,
		};
	}
}