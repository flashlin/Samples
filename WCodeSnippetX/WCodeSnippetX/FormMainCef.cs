using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CefSharp.WinForms;

namespace WCodeSnippetX
{
	public partial class FormMainCef : Form
	{
		public FormMainCef()
		{
			InitializeComponent();
			Initialize();
		}

		private void Initialize()
		{
			var fileName = Path.Combine(Directory.GetCurrentDirectory(), "Views/index.html");
			var browser = new ChromiumWebBrowser(fileName)
			{
				Dock = DockStyle.Fill
			};
			this.Controls.Add(browser);
		}
	}
}
