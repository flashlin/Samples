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

    public bool IsEmpty => Width == 0 && Height == 0;

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

    public static Rect OfSize(Size size) => new Rect
    {
        Left = 0, 
        Top = 0, 
        Width = size.Width, 
        Height = size.Height
    };

    public bool Contain(Position pos)
    {
        return (pos.X >= Left && pos.X <= Right &&
                pos.Y >= Top && pos.Y <= Bottom);
    }
}