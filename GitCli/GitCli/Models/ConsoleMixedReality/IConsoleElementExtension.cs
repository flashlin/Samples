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
}