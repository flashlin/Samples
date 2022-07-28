namespace T1.ConsoleUiMixedReality;

public struct Size
{
    public int Width { get; set; }
    public int Height { get; set; }

    public override bool Equals(object? obj)
    {
        if (obj == null)
        {
            return false;
        }

        if (obj is Rect b)
        {
            return Width == b.Width && Height == b.Height;
        }

        return false;
    }

    public static bool operator ==(Size a, Size b) => a.Width == b.Width && a.Height == b.Height;
    public static bool operator !=(Size a, Size b) => !(a == b);
    public static bool operator <=(Size a, Size b) => a.Width <= b.Width && a.Height <= b.Height;
    public static bool operator >=(Size a, Size b) => a.Width >= b.Width && a.Height >= b.Height;

    public override int GetHashCode()
    {
        return HashCodeCalculator.GetHashCode(
            (typeof(int), Width),
            (typeof(int), Height)
        );
    }
}