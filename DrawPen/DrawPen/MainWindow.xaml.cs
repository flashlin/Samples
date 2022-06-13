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
using System.Windows.Shapes;

namespace DrawPen
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		bool _start = false;
		private DrawWindow _drawWindow;

		public MainWindow()
		{
			InitializeComponent();
		}

		private void handleClickStart(object sender, RoutedEventArgs e)
		{
			ToggleDrawWindow();
		}

		private void ToggleDrawWindow()
		{
			_start = !_start;

			if (_start)
			{
				BtnStart.SetValue(ContentProperty, "Stop");
				_drawWindow = new DrawWindow();
				_drawWindow.Show();
			} 
			else
			{
				BtnStart.SetValue(ContentProperty, "Start");
				_drawWindow.Close();
			}
		}
	}
}
