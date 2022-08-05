using CommunityToolkit.Maui.Markup;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CommunityToolkit.Maui.Markup.GridRowsColumns;

namespace MaCodeSnippet.Views
{
	public class MainDataTemplate : DataTemplate
	{
		public MainDataTemplate() : base(CreateGrid)
		{

		}

		static Grid CreateGrid() => new()
		{
			RowSpacing = 1,

			RowDefinitions = Rows.Define(
				(Row.Title, 20),
				(Row.Description, 20),
				(Row.BottomPadding, 1)),

			Children =
			{
				new Label()
					.Row(Row.Title).Top()
					.Font(size: 16).TextColor(ColorConstants.TextCellTextColor)
					.Paddings(10, 0, 10, 0)
					.Bind(Label.TextProperty, "Index"),

				new Label()
					.Row(Row.Description)
					.Font(size: 13).TextColor(ColorConstants.TextCellDetailColor)
					.Bind(Label.TextProperty, "Context")
			}
		};

		enum Row { Title, Description, BottomPadding }
	}


	public static class ColorConstants
	{
		public static Color NavigationBarBackgroundColor { get; } = Color.FromArgb("FF6601");
		public static Color NavigationBarTextColor { get; } = Colors.Black;

		public static Color TextCellDetailColor { get; } = Color.FromArgb("828282");
		public static Color TextCellTextColor { get; } = Colors.Black;

		public static Color BrowserNavigationBarBackgroundColor { get; } = Color.FromArgb("FFE6D5");
		public static Color BrowserNavigationBarTextColor { get; } = Color.FromArgb("3F3F3F");
	}
}
