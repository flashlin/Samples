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
}