using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Terminal.Gui;
using NStack;
using GitCli.Models;


namespace GitCli.Models
{
	public class Main
	{
		public Task Run()
		{
			Application.Init();
			var top = Application.Top;

			var win = new Window("MyApp")
			{
				X = 0,
				Y = 1, // Leave one row for the toplevel menu
						 // By using Dim.Fill(), it will automatically resize without manual intervention
				Width = Dim.Fill(),
				Height = Dim.Fill()
			};
			top.Add(win);


			var menu = new MenuBar(new MenuBarItem[] {
				new MenuBarItem ("_File", new MenuItem [] {
					new MenuItem ("Clone...", "Clone Repository", null),
					new MenuItem ("_Open Repository...", "Open Repository",null),
					new MenuItem ("_Exit", "", () => { if (Quit ()) top.Running = false; })
				}),
				new MenuBarItem ("_View", new MenuItem [] {
					new MenuItem ("Show Uncommitted Changes", "", null),
				}),
				new MenuBarItem ("_Repository", new MenuItem [] {
					new MenuItem ("Refresh", "", null),
					new MenuItem ("Fetch...", "", null),
					new MenuItem ("Pull...", "", null),
					new MenuItem ("Push...", "", null),
					new MenuItem ("Save Stash...", "", null),
					new MenuItem ("New Branch...", "", null),
					new MenuItem ("Rebase Merge...", "", null),
				}),
			});
			top.Add(menu);

			Application.Run();
			Application.Shutdown();

			return Task.CompletedTask;
		}

		static bool Quit()
		{
			var n = MessageBox.Query(50, 7, "Quit GitCli", "Are you sure you want to quit this GitCli?", "Yes", "No");
			return n == 0;
		}

		static void OpenRepository()
		{

		}


	}
}
