namespace GitCli.Models.ConsoleMixedReality;

public static class ConsoleElementExtension
{
	public static T Setup<T>(this T elem, Action<T> setupFn)
		 where T : IConsoleElement
	{
		setupFn(elem);
		return elem;
	}

	public static IConsoleElement GetLeafChild(this IConsoleElement element)
	{
		if (element.Children.Count != 0)
		{
			return element.Children.GetFocusedControl().GetLeafChild();
		}
		return element;
	}

	public static void HandleOnCreate(this IConsoleElement element, Rect rect, IConsoleManager consoleManager)
	{
		element.ConsoleManager = consoleManager;
		element.Children.ConsoleManager = consoleManager;
		element.ViewRect = element.DesignRect.ToViewRect(rect, consoleManager);
		consoleManager.FirstSetFocusElement(element);
	}

	public static bool RaiseOnBubbleEvent(this IConsoleElement? parent, IConsoleElement element, InputEvent inputEvent)
	{
		return parent?.OnBubbleEvent(element, inputEvent) ?? false;
	}

	public static int GetDesignRectWidthOrViewWidth(this IConsoleElement element)
	{
		var childDesignWidth = element.Children.Max(x => x.DesignRect.Width);
		var designWidth = Math.Max(element.DesignRect.Width, childDesignWidth);
		var lastDesignWidth = (designWidth == 0 ? element.ViewRect.Width : designWidth);
		if (element.ViewRect.Width == 0)
		{
			return lastDesignWidth;
		}
		return Math.Min(lastDesignWidth, element.ViewRect.Width);
	}
}