namespace GitCli.Models.ConsoleMixedReality;

public struct Rect
{
    public static Rect Empty = new Rect
    {
        Left = 0,
        Top = 0,
        Width = 0,
        Height = 0
    };

    public int Left { get; init; }
    public int Top { get; init; }
    public int Right => Left + Width - 1;

    public int Bottom => Top + Height - 1;

    public int Width { get; init; }
    public int Height { get; init; }

    public Position TopLeftCorner => new Position(Left, Top);

    public Position BottomRightCorner => new Position(Right, Bottom);

    public bool IsEmpty => Width == 0 && Height == 0;

    public Rect Move(int x, int y)
    {
        return new Rect
        {
            Left = Left + x,
            Top = Top + y,
            Width = Width,
            Height = Height
        };
    }

    public override string ToString()
    {
        return $"{Left},{Top},{Width},{Height}";
    }

    public Rect Intersect(Rect b)
    {
        if (this.IsEmpty || b.IsEmpty) return Empty;
        return new Rect
        {
            Left = Math.Max(this.Left, b.Left),
            Top = Math.Max(this.Top, b.Top),
            Width = Math.Min(this.Right, b.Right) - Math.Max(this.Left, b.Left) + 1,
            Height = Math.Min(this.Bottom, b.Bottom) - Math.Max(this.Top, b.Top) + 1
        };
    }


    public Rect Init(Func<Rect> initSizeFn)
	 {
		if (!IsEmpty)
		{
         return this;
		}
      return initSizeFn();
	 }

    public static Rect OfSize(Size size) => new Rect
    {
        Left = 0,
        Top = 0,
        Width = size.Width,
        Height = size.Height
    };

    public static Rect Of(Position position) => new Rect
    {
        Left = position.X,
        Top = position.Y,
        Width = 1,
        Height = 1
    };

    public Rect ExtendBy(Position position)
    {
        if (IsEmpty) return Of(position);

        var left = Math.Min(Left, position.X);
        var top = Math.Min(Top, position.Y);
        var width = Math.Max(Right, position.X) - left + 1;
        var height = Math.Max(Bottom, position.Y) - top + 1;

        return new Rect
        {
            Left = left,
            Top = top,
            Width = width,
            Height = height
        };
    }

    public Rect Surround(Rect b)
    {
        if (b.IsEmpty) return this;
        return ExtendBy(b.TopLeftCorner)
            .ExtendBy(b.BottomRightCorner);
    }


    public bool Contain(Position pos)
    {
        return (pos.X >= Left && pos.X <= Right &&
                pos.Y >= Top && pos.Y <= Bottom);
    }
}