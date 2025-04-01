using System;

namespace VimSharpLib;

public class ColoredCharScreen
{
    private readonly ColoredChar[,] _screen;

    public ColoredCharScreen(int height, int width)
    {
        _screen = new ColoredChar[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                _screen[y, x] = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
            }
        }
    }

    public ColoredChar this[int y, int x]
    {
        get => _screen[y, x];
        set => _screen[y, x] = value;
    }

    public int GetLength(int dimension)
    {
        return _screen.GetLength(dimension);
    }
} 