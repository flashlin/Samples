using System;
using System.Linq;

namespace T1.ConsoleUiMixedReality;

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

	public static bool RaiseOnBubbleKeyEvent(this IConsoleElement? parent, IConsoleElement element, InputEvent inputEvent)
	{
		return parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
	}

	public static bool RaiseOnBubbleEvent(this IConsoleElement? parent, IConsoleElement element, ConsoleElementEvent evt)
	{
		return parent?.RaiseOnBubbleEvent(element, evt) ?? false;
	}

	public static int GetDesignRectValue(this IConsoleElement element, Func<Rect, int> getRectValue)
	{
		var childDesignValue = 0;
		if (element.Children.Count > 0)
		{
			childDesignValue = element.Children.Max(x => x.GetDesignRectValue(getRectValue));
		}
		return Math.Max(getRectValue(element.DesignRect), childDesignValue);
	}

	public static int GetDesignRectOrViewValue(this IConsoleElement element, Func<Rect, int> getRectValue)
	{
		var designValue = GetDesignRectValue(element, getRectValue);
		var lastDesignValue = (designValue == 0 ? getRectValue(element.ViewRect) : designValue);
		if (element.ViewRect.Width == 0)
		{
			return lastDesignValue;
		}
		return Math.Min(lastDesignValue, getRectValue(element.ViewRect));
	}
}