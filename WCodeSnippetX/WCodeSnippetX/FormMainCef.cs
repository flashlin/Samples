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
		private readonly GlobalKeyboardHook _globalKeyboardHook = new();
		private readonly IBoundObject _boundObject;
		bool _leftAlt = false;

		public FormMainCef(IBoundObject boundObject)
		{
			_boundObject = boundObject;
			_boundObject.Form = this;
			InitializeComponent();
			Initialize();
			this.Activated += FormMainCef_Activated;
			_globalKeyboardHook.KeyboardPressed += OnGlobalKeyPressed;
		}

		private void OnGlobalKeyPressed(object? sender, GlobalKeyboardHookEventArgs e)
		{
			if (e.KeyboardState != GlobalKeyboardHook.KeyboardState.KeyDown &&
			    e.KeyboardState != GlobalKeyboardHook.KeyboardState.SysKeyDown)
			{
				_leftAlt = false;
				return;
			}

			if (e.KeyboardData.VirtualCode == VirtualCode.VkLeftAlt)
			{
				_leftAlt = true;
				e.Handled = true;
				return;
			}

			if (e.KeyboardData.VirtualCode == VirtualCode.Vk0 && _leftAlt)
			{
				BringMeToFront();
				e.Handled = true;
				return;
			}
		}


		private void FormMainCef_Activated(object? sender, EventArgs e)
		{
			var screenWidth = SystemInformation.PrimaryMonitorSize.Width;
			var screenHeight = SystemInformation.PrimaryMonitorSize.Height;
			this.Width = (int)(screenWidth * 0.3);
			this.Height = (int)(screenHeight * 0.6);
			this.Left = screenWidth - this.Width - 30;
			this.Top = screenHeight - this.Height - 60;
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
			Invoke(() =>
			{
				this.WindowState = FormWindowState.Minimized;
			});
		}

		public void BringMeToFront()
		{
			this.WindowState = FormWindowState.Minimized;
			this.Show();
			this.WindowState = FormWindowState.Normal;
		}
	}
}
