namespace GitCli.Models.ConsoleMixedReality;

public struct Size
{
    public int Width { get; init; }
    public int Height { get; init; }

    public static bool operator ==(Size a, Size b) => a.Width == b.Width && a.Height == b.Height;
    public static bool operator !=(Size a, Size b) => !(a == b);
    public static bool operator <=(Size a, Size b) => a.Width <= b.Width && a.Height <= b.Height;
    public static bool operator >=(Size a, Size b) => a.Width >= b.Width && a.Height >= b.Height;
}