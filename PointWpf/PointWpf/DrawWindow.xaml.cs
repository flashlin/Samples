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
using System.Windows.Shapes;
using Colors = Microsoft.Maui.Graphics.Colors;

namespace PointWpf
{
	/// <summary>
	/// Interaction logic for DrawWindow.xaml
	/// </summary>
	public partial class DrawWindow : Window
	{
		bool _painting = false;
		List<DrawPoint> _points = new List<DrawPoint>();
		private readonly MainWindow parent;

		public DrawWindow(MainWindow parent)
		{
			InitializeComponent();
			this.parent = parent;
		}

		private void handleKeyDown(object sender, KeyEventArgs e)
		{
			if (e.Key == Key.F9)
			{
				_points.Clear();
				DrawSurface.InvalidateVisual();
				parent.CloseDrawWindow();
			}
		}

		private void SKElement_PaintSurface(object sender, SkiaSharp.Views.Desktop.SKPaintSurfaceEventArgs e)
		{
			ICanvas canvas = new SkiaCanvas()
			{
				Canvas = e.Surface.Canvas
			};

			//canvas.FillColor = Colors.Transparent;
			//canvas.FillColor = Colors.White.WithAlpha(0.9f);
			//canvas.FillRectangle(0, 0, (float)DrawSurface.ActualWidth, (float)DrawSurface.ActualHeight);


			//canvas.StrokeColor = Colors.Red.WithAlpha(.5f);
			canvas.StrokeColor = Colors.Red;
			canvas.StrokeSize = 2;
			var path = new PathF();
			if (_points.Count > 0)
			{
				path.MoveTo(_points[0].X, _points[0].Y);
				for (int i = 1; i < _points.Count; i++)
				{
					path.LineTo(_points[i].X, _points[i].Y);
				}
				//path.Close();
				canvas.DrawPath(path);
			}
		}

		private void handleMouseDown(object sender, MouseButtonEventArgs e)
		{
			_painting = true;
		}

		private void handleMouseUp(object sender, MouseButtonEventArgs e)
		{
			_painting = false;
		}

		private void SKElement_MouseMove(object sender, MouseEventArgs e)
		{
			var p = e.GetPosition(DrawSurface);
			if (_painting)
			{
				_points.Add(new DrawPoint { X = (float)p.X, Y = (float)p.Y });
				DrawSurface.InvalidateVisual();
			}
		}

		private void SKElement_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
		{
			_painting = true;
		}

		private void SKElement_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
		{
			_painting = false;
		}
	}
}
