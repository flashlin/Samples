using ExpectedObjects;
using GitCli.Models.ConsoleMixedReality;

namespace GitCliTest;

[TestFixture]
public class StrSpanTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void BSpanInLeft()
    {
        var aSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var bSpan = new StrSpan
        {
            Index = -5,
            Length = 10
        };

        var actualResult = aSpan.NonIntersect(bSpan)
            .ToArray();

        new[]
        {
            new StrSpan
            {
                Index = -5,
                Length = 5
            },
        }.ToExpectedObject().ShouldEqual(actualResult);
    }

    [Test]
    public void BSpanInRight()
    {
        var aSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var bSpan = new StrSpan
        {
            Index = 5,
            Length = 10
        };

        var actualResult = aSpan.NonIntersect(bSpan)
            .ToArray();

        new[]
        {
            new StrSpan
            {
                Index = 10,
                Length = 5
            },
        }.ToExpectedObject().ShouldEqual(actualResult);
    }

    [Test]
    public void BSpanInLarge()
    {
        var aSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var bSpan = new StrSpan
        {
            Index = -5,
            Length = 20
        };

        var actualResult = aSpan.NonIntersect(bSpan)
            .ToArray();

        new[]
        {
            new StrSpan
            {
                Index = -5,
                Length = 5
            },
            new StrSpan
            {
                Index = 10,
                Length = 5
            },
        }.ToExpectedObject().ShouldEqual(actualResult);
    }
    
    
    [Test]
    public void BSpanSame()
    {
        var aSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var bSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var actualResult = aSpan.NonIntersect(bSpan)
            .ToArray();

        Assert.That(actualResult, Is.Empty);
    }
    
    
    [Test]
    public void NoIntersect()
    {
        var aSpan = new StrSpan
        {
            Index = 0,
            Length = 10
        };

        var bSpan = new StrSpan
        {
            Index = 10,
            Length = 10
        };

        var actualResult = aSpan.NonIntersect(bSpan)
            .ToArray();

        Assert.That(actualResult, Is.Empty);
    }
}