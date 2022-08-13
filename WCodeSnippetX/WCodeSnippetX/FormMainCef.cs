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
		private readonly IBoundObject _boundObject;

		public FormMainCef(IBoundObject boundObject)
		{
			_boundObject = boundObject;
			_boundObject.Form = this;
			InitializeComponent();
			Initialize();
			this.Activated += FormMainCef_Activated;
		}

		private void FormMainCef_Activated(object? sender, EventArgs e)
		{
			this.Width = (int)(SystemInformation.VirtualScreen.Width * 0.3);
			this.Height = (int)(SystemInformation.VirtualScreen.Height * 0.6);
			this.Left =  SystemInformation.VirtualScreen.Width - this.Width - 30;
			this.Top = SystemInformation.VirtualScreen.Height - this.Height - 60;
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

		public void Minimize()
		{
			this.WindowState = FormWindowState.Minimized;
		}

		public void BringMeToFront()
		{
			this.WindowState = FormWindowState.Minimized;
			this.Show();
			this.WindowState = FormWindowState.Normal;
		}
	}
}
