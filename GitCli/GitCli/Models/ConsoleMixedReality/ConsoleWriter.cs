namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleWriter : IConsoleWriter
{
    private readonly ConsoleColor _originBackgroundColor;
    private readonly ConsoleColor _originForegroundColor;
    private ConsoleColor _foregroundColor;
    private ConsoleColor _backgroundColor;

    public ConsoleWriter()
    {
        _originBackgroundColor = Console.BackgroundColor;
        _backgroundColor = _originBackgroundColor;
        _originForegroundColor = Console.ForegroundColor;
        _foregroundColor = _originForegroundColor;
    }

    public ConsoleSize GetSize()
    {
        return new ConsoleSize()
        {
            Width = Console.WindowWidth,
            Height = Console.WindowHeight,
        };
    }

    public void SetForegroundColor(ConsoleColor color)
    {
        _foregroundColor = color;
        Console.ForegroundColor = color;
    }

    public void SetBackgroundColor(ConsoleColor color)
    {
        _backgroundColor = color;
        Console.BackgroundColor = color;
    }

    public void Write(string text)
    {
        Console.Write(text);
    }

    public void WriteLine(string text)
    {
        Console.WriteLine(text);
    }

    public void ResetWriteColor()
    {
        Console.BackgroundColor = _backgroundColor;
        Console.ForegroundColor = _foregroundColor;
    }

    public void ResetColor()
    {
        Console.BackgroundColor = _originBackgroundColor;
        Console.ForegroundColor = _originForegroundColor;
    }
}

public struct ConsoleLocation
{
    public int X { get; set; }
    public int Y { get; set; }

    public bool XInRight(ConsoleRectangle rectangle)
    {
        return X > rectangle.RightBottom.X;
    }

    public bool YInRight(ConsoleRectangle rectangle)
    {
        return Y >= rectangle.LeftTop.Y &&
               Y <= rectangle.LeftTop.Y;
    }
}

public struct ConsoleRectangle
{
    public static ConsoleRectangle Empty = new ConsoleRectangle
    {
        LeftTop = new ConsoleLocation
        {
            X = 0,
            Y = 0,
        },
        RightBottom = new ConsoleLocation()
        {
            X = 0,
            Y = 0,
        }
    };

    public ConsoleLocation LeftTop { get; set; }
    public ConsoleLocation RightBottom { get; set; }

    public ConsoleLocation GetRightTop()
    {
        return new ConsoleLocation
        {
            X = RightBottom.X,
            Y = LeftTop.Y
        };
    }

    public ConsoleLocation GetLeftBottom()
    {
        return new ConsoleLocation
        {
            X = LeftTop.X,
            Y = RightBottom.Y
        };
    }


    public int GetWidth()
    {
        return RightBottom.X - LeftTop.X + 1;
    }

    public int GetHeight()
    {
        return RightBottom.Y - LeftTop.Y + 1;
    }

    public bool Contain(ConsoleLocation point)
    {
        return ContainX(point) && ContainY(point);
    }

    private bool ContainY(ConsoleLocation point)
    {
        return point.Y >= LeftTop.Y && point.Y <= RightBottom.Y;
    }

    private bool ContainX(ConsoleLocation point)
    {
        return point.X >= LeftTop.X && point.X <= RightBottom.X;
    }

    public bool IsTop(ConsoleLocation point)
    {
        return point.Y < LeftTop.Y;
    }

    public bool IsRight(ConsoleLocation point)
    {
        return point.X > RightBottom.X;
    }

    public bool IsBottom(ConsoleLocation point)
    {
        return point.Y > RightBottom.Y;
    }

    public bool IsLeft(ConsoleLocation point)
    {
        return point.X < LeftTop.X;
    }

    public ConsoleRectangle Intersect(ConsoleRectangle b)
    {
        if (InTop(b))
        {
            return new ConsoleRectangle()
            {
                LeftTop = new ConsoleLocation
                {
                    X = b.LeftTop.X,
                    Y = LeftTop.Y,
                },
                RightBottom = b.RightBottom,
            };
        }
        
        if (InRightTop(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = b.LeftTop.X,
                    Y = LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = RightBottom.X,
                    Y = b.RightBottom.Y
                }
            };
        }

        if (InRight(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = b.LeftTop.X,
                    Y = b.LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = RightBottom.X,
                    Y = b.RightBottom.Y,
                }
            };
        }

        if (InRightBottom(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = b.LeftTop.X,
                    Y = b.LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = RightBottom.X,
                    Y = RightBottom.Y,
                },
            };
        }

        if (InBottom(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = b.LeftTop.X,
                    Y = b.LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = b.RightBottom.X,
                    Y = RightBottom.Y,
                }
            };
        }

        if (InLeftBottom(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = LeftTop.X,
                    Y = b.LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = b.RightBottom.X,
                    Y = RightBottom.Y,
                }
            };
        }

        if (IsLeft(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = LeftTop.X,
                    Y = b.LeftTop.Y,
                },
                RightBottom = new ConsoleLocation
                {
                    X = b.RightBottom.X,
                    Y = b.RightBottom.Y
                },
            };
        }

        if (IsLeftTop(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = new ConsoleLocation
                {
                    X = LeftTop.X,
                    Y = LeftTop.Y,
                },
                RightBottom = b.RightBottom
            };
        }

        if (IsIn(b))
        {
            return new ConsoleRectangle
            {
                LeftTop = b.LeftTop,
                RightBottom = b.RightBottom
            };
        }
        
        
        if (IsOut(b))
        {
            return this;
        }

        return ConsoleRectangle.Empty;
    }

    private bool IsOut(ConsoleRectangle b)
    {
        return b.Contain(LeftTop) &&
               b.Contain(RightBottom);
    }

    private bool IsIn(ConsoleRectangle b)
    {
        return Contain(b.LeftTop) &&
               Contain(b.RightBottom);
    }

    private bool IsLeftTop(ConsoleRectangle b)
    {
        return Contain(b.RightBottom) &&
               IsLeft(b.LeftTop) &&
               IsLeft(b.GetLeftBottom()) &&
               IsTop(b.GetRightTop());
    }

    private bool IsLeft(ConsoleRectangle b)
    {
        return Contain(b.GetRightTop()) &&
               Contain(b.RightBottom) &&
               !Contain(b.LeftTop) &&
               !Contain(b.GetLeftBottom());
    }

    private bool InLeftBottom(ConsoleRectangle b)
    {
        return IsLeft(b.LeftTop) &&
               Contain(b.GetRightTop()) &&
               !Contain(b.GetLeftBottom()) &&
               !Contain(b.RightBottom);
    }

    private bool InBottom(ConsoleRectangle b)
    {
        return Contain(b.LeftTop) &&
               Contain(b.GetRightTop()) &&
               IsBottom(b.GetLeftBottom()) &&
               IsBottom(b.RightBottom);
    }

    private bool InRightBottom(ConsoleRectangle b)
    {
        return Contain(b.LeftTop) &&
               !Contain(b.GetRightTop()) &&
               !Contain(b.RightBottom) &&
               !Contain(b.GetLeftBottom());
    }

    private bool InRight(ConsoleRectangle b)
    {
        return Contain(b.LeftTop) &&
               Contain(b.GetLeftBottom()) &&
               IsRight(b.GetRightTop()) &&
               IsRight(b.RightBottom);
    }

    private bool InRightTop(ConsoleRectangle b)
    {
        return b.Contain(GetRightTop()) &&
               Contain(b.GetLeftBottom());
    }

    private bool InTop(ConsoleRectangle b)
    {
        return IsTop(b.LeftTop) &&
               IsTop(b.GetRightTop()) &&
               Contain(b.GetLeftBottom()) &&
               Contain(b.RightBottom);
    }
}

public class ConsoleTextBox : IConsoleElement
{
    public int EditIndex { get; set; }
    public string Value { get; set; }
    public ConsoleLocation Location { get; set; }
    public ConsoleSize Size { get; set; }

    public void Show(ConsoleRectangle rectangle)
    {
        var console = new ConsoleWriter();

        var x = rectangle.LeftTop.X + Location.X;
        var y = rectangle.LeftTop.Y + Location.Y;
    }
}

public interface IConsoleElement
{
}

public class ConsoleView
{
    private ConsoleSize _size;

    public void SetSize(ConsoleSize size)
    {
        _size = size;
    }

    public void Show()
    {
    }
}