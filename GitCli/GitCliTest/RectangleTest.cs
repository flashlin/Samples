using ExpectedObjects;
using T1.ConsoleUiMixedReality;

namespace GitCliTest;

public class RectangleTest
{
    private Rect _aRect = new Rect
    {
        Left = 0,
        Top = 0,
        Width = 10,
        Height = 10
    };

    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Top()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = -2,
            Width = 4,
            Height = 4
        });

        new Rect
        {
            Left = 2,
            Top = 0,
            Width = 4,
            Height = 2
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void RightTop()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = -2,
            Width = 10,
            Height = 5
        });

        new Rect
        {
            Left = 2,
            Top = 0,
            Width = 8,
            Height = 3
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void Right()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = 2,
            Width = 15,
            Height = 5
        });

        new Rect
        {
            Left = 2,
            Top = 2,
            Width = 8,
            Height = 5
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void RightBottom()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = 2,
            Width = 15,
            Height = 15
        });

        new Rect
        {
            Left = 2,
            Top = 2,
            Width = 8,
            Height = 8
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void Bottom()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = 2,
            Width = 5,
            Height = 15
        });

        new Rect
        {
            Left = 2,
            Top = 2,
            Width = 5,
            Height = 8
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void LeftBottom()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = -2,
            Top = 2,
            Width = 5,
            Height = 15
        });

        new Rect
        {
            Left = 0,
            Top = 2,
            Width = 3,
            Height = 8
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void Left()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = -2,
            Top = 2,
            Width = 5,
            Height = 8
        });

        new Rect
        {
            Left = 0,
            Top = 2,
            Width = 3,
            Height = 8
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void LeftTop()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = -2,
            Top = -2,
            Width = 5,
            Height = 8
        });

        new Rect
        {
            Left = 0,
            Top = 0,
            Width = 3,
            Height = 6
        }.ToExpectedObject().ShouldEqual(rect);
    }


    [Test]
    public void In()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = 2,
            Top = 2,
            Width = 5,
            Height = 5
        });

        new Rect
        {
            Left = 2,
            Top = 2,
            Width = 5,
            Height = 5
        }.ToExpectedObject().ShouldEqual(rect);
    }

    [Test]
    public void Out()
    {
        var rect = _aRect.Intersect(new Rect
        {
            Left = -2,
            Top = -2,
            Width = 15,
            Height = 15
        });

        new Rect
        {
            Left = 0,
            Top = 0,
            Width = 10,
            Height = 10
        }.ToExpectedObject().ShouldEqual(rect);
    }
}