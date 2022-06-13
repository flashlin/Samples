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

namespace DrawPen
{
	/// <summary>
	/// Interaction logic for DrawWindow.xaml
	/// </summary>
	public partial class DrawWindow : Window
	{
		private bool _painting;
		private List<DrawPoint> _points = new List<DrawPoint>();

		public DrawWindow()
		{
			InitializeComponent();
		}

		private void SKElement_PaintSurface(object sender, SkiaSharp.Views.Desktop.SKPaintSurfaceEventArgs e)
		{
			ICanvas canvas = new SkiaCanvas()
			{
				Canvas = e.Surface.Canvas
			};
			
			canvas.StrokeColor = Colors.Red;
			canvas.StrokeSize = 2;
			//draw many points
			var path = new PathF();
			if(_points.Count > 0 )
			{
				path.MoveTo(_points[0].X, _points[0].Y);
				for (int i = 1; i < _points.Count; i++)
				{
					path.LineTo(_points[i].X, _points[i].Y);
				}
				canvas.DrawPath(path);
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

		private void SKElement_MouseMove(object sender, MouseEventArgs e)
		{
			var point = e.GetPosition(DrawSurface);
			if (_painting)
			{
				_points.Add(new DrawPoint { X = (float)point.X, Y = (float)point.Y });
				DrawSurface.InvalidateVisual();
			}
		}
	}
}
