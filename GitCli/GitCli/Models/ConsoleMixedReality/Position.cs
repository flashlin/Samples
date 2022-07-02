namespace GitCli.Models.ConsoleMixedReality;

public struct Position
{
    public Position(int x, int y)
    {
        X = x;
        Y = y;
    }

    public int X { get; init; }
    public int Y { get; init; }

   public Position Next => new Position(X + 1, Y);


	public static bool operator ==(in Position lhs, in Position rhs) => lhs.X == rhs.X && lhs.Y == rhs.Y;
	public static bool operator !=(in Position lhs, in Position rhs) => !(lhs == rhs);

	public override bool Equals(object obj) => obj is Position position && this == position;

	public override int GetHashCode()
	{
		var hashCode = -695327075;
		hashCode = hashCode * -1521134295 + X.GetHashCode();
		hashCode = hashCode * -1521134295 + Y.GetHashCode();
		return hashCode;
	}
}