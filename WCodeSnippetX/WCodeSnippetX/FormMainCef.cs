using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CefSharp;
using CefSharp.Internals;
using CefSharp.WinForms;
using WCodeSnippetX.Models;

namespace WCodeSnippetX
{
	public partial class FormMainCef : Form
	{
		private IBoundObject _boundObject;

		public FormMainCef(IBoundObject boundObject)
		{
			_boundObject = boundObject;
			InitializeComponent();
			Initialize();
		}

		private void Initialize()
		{
			//var fileName = Path.Combine(Directory.GetCurrentDirectory(), "Views/index.html");
			var fileName = "localfolder://cefsharp/";
			var browser = new ChromiumWebBrowser(fileName)
			{
				Dock = DockStyle.Fill
			};
			browser.JavascriptObjectRepository.Settings.LegacyBindingEnabled = true;
			browser.JavascriptObjectRepository.Register("__backend", _boundObject);
			//browser.JavascriptObjectRepository.ResolveObject += (s, e) =>
			//{
			//	if (e.ObjectName == JavascriptObjectRepository.LegacyObjects)
			//	{
			//		e.ObjectRepository.Register("__backend", obj);
			//	}
			//};

			this.Controls.Add(browser);
			//browser.IsBrowserInitializedChanged += BrowserOnIsBrowserInitializedChanged;
		}

		bool _isOpened = false;

		private void BrowserOnIsBrowserInitializedChanged(object? sender, EventArgs e)
		{
			var browser = (ChromiumWebBrowser)sender!;
			if (browser.IsBrowserInitialized && !_isOpened)
			{
				_isOpened = true;
				browser.ShowDevTools();
			}
		}
	}
}
