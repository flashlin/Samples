using ExpectedObjects;
using GitCli.Models.ConsoleMixedReality;

namespace GitCliTest;

public class RectangleTest
{
	private ConsoleRectangle _aRect = new ConsoleRectangle
	{
		LeftTop = new ConsoleLocation
		{
			X = 0,
			Y = 0,
		},
		RightBottom = new ConsoleLocation
		{
			X = 10,
			Y = 10
		},
	};

	[SetUp]
	public void Setup()
	{
	}

	[Test]
	public void Top()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = -2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 4,
				Y = 4
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 0,
			},
			RightBottom = new ConsoleLocation
			{
				X = 4,
				Y = 4
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void RightTop()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = -2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 14,
				Y = 4
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 0,
			},
			RightBottom = new ConsoleLocation
			{
				X = 10,
				Y = 4
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void Right()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 15,
				Y = 5
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 10,
				Y = 5
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void RightBottom()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 15,
				Y = 15
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 10,
				Y = 10
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void Bottom()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 15
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 10
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void LeftBottom()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = -2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 15
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 0,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 10
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void Left()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = -2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 8
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 0,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 8
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void LeftTop()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = -2,
				Y = -2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 8
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 0,
				Y = 0,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 8
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void In()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 5
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 2,
				Y = 2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 5,
				Y = 5
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
	[Test]
	public void Out()
	{
		var rect = _aRect.Intersect(new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = -2,
				Y = -2,
			},
			RightBottom = new ConsoleLocation
			{
				X = 15,
				Y = 15
			}
		});

		new ConsoleRectangle
		{
			LeftTop = new ConsoleLocation
			{
				X = 0,
				Y = 0,
			},
			RightBottom = new ConsoleLocation
			{
				X = 10,
				Y = 10
			}
		}.ToExpectedObject().ShouldEqual(rect);
	}
	
	
}