namespace T1.ConsoleUiMixedReality;

public struct Position
{
	public Position(int x, int y)
	{
		X = x;
		Y = y;
	}

	public static Position Empty = new Position(-1, -1)
	{
		IsEmpty = true,
	};

	public bool IsEmpty { get; private set; } = false;
	public int X { get; set; } = 0;
	public int Y { get; set; } = 0;

	public Position Next => new Position(X + 1, Y);

	public static bool operator ==(in Position lhs, in Position rhs) => lhs.X == rhs.X && lhs.Y == rhs.Y;
	public static bool operator !=(in Position lhs, in Position rhs) => !(lhs == rhs);

	public override bool Equals(object? obj) => obj is Position position && this == position;

	public override int GetHashCode()
	{
		var hashCode = -695327075;
		hashCode = hashCode * -1521134295 + X.GetHashCode();
		hashCode = hashCode * -1521134295 + Y.GetHashCode();
		return hashCode;
	}

	public override string ToString()
	{
		return $"{X},{Y}";
	}
}