using Microsoft.Maui.Graphics;
using Microsoft.Maui.Graphics.Skia;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using Colors = Microsoft.Maui.Graphics.Colors;

namespace PointWpf
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		bool _start = false;
		DrawWindow? _drawWindow;

		public MainWindow()
		{
			InitializeComponent();
		}

		private void handleKeyDown(object sender, KeyEventArgs e)
		{
			if (e.Key != Key.F9)
			{
				return;
			}

			ToggleDrawWindow();
		}

		public void OpenDrawWindow()
		{
			_start = true;
			BtnStart.SetValue(Button.ContentProperty, "Close");
			_drawWindow = new DrawWindow(this);
			_drawWindow.Show();
		}

		public void CloseDrawWindow()
		{
			_start = false;
			BtnStart.SetValue(Button.ContentProperty, "Start");
			_drawWindow?.Close();
		}

		private void handleClickStart(object sender, RoutedEventArgs e)
		{
			ToggleDrawWindow();
		}

		private void ToggleDrawWindow()
		{
			if (_start)
			{
				CloseDrawWindow();
			}
			else
			{
				OpenDrawWindow();
			}
		}

		private void handleClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			CloseDrawWindow();
		}
	}
}
