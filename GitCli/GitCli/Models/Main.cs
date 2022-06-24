using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Terminal.Gui;
using NStack;
using GitCli.Models;
using LibGit2Sharp;

namespace GitCli.Models
{
	public class Main
	{
		public Task Run()
		{
			new Example().Sample();


			Application.Init();
			var top = Application.Top;

			var workspace = new Window("Workspace")
			{
				X = 0,
				Y = 1,
				Width = Dim.Percent(30),
				Height = Dim.Percent(30)
			};
			top.Add(workspace);
			AddWorkSpaceMenu(workspace);


			workspace.GetCurrentHeight(out var workspaceHeight);
			var repositoryWin = new Window("Repository")
			{
				X = 0,
				Y = workspaceHeight + 1,
				Width = Dim.Fill(),
				Height = Dim.Fill()
			};
			top.Add(repositoryWin);

			workspace.GetCurrentWidth(out var workspaceWidth);
			var unstagedWin = new Window("unstaged")
			{
				X = workspace.X + workspaceWidth + 1,
				Y = 1,
				Width = Dim.Fill(),
				Height = Dim.Fill()
			};
			top.Add(unstagedWin);

			AddMenuBar(top);

			Application.Run();
			Application.Shutdown();

			return Task.CompletedTask;
		}

		private void AddMenuBar(Toplevel top)
		{
			var menu = new MenuBar(new MenuBarItem[] {
				new MenuBarItem ("_File", new MenuItem [] {
					new MenuItem ("Clone...", "Clone Repository", null),
					new MenuItem ("_Open Repository...", "Open Repository", OpenRepository),
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
		}

		private void AddWorkSpaceMenu(Window workspace)
		{
			var menus = new List<string>()
			{
				"Changes",
				"All Commits"
			};
			var workspaceView = new ListView(menus)
			{
				X = 0,
				Y = 0,
				Width = Dim.Fill(),
				Height = Dim.Fill(),
			};
			workspace.Add(workspaceView);

			workspaceView.OpenSelectedItem += WorkspaceView_OpenSelectedItem; 
		}

		private void WorkspaceView_OpenSelectedItem(ListViewItemEventArgs obj)
		{
			if( obj.Item == 0 )
			{
				Confirm("Changes", "123");
			}
		}

		bool Confirm(string title, string message)
		{
			var n = MessageBox.Query(50, 7, title, message, "Yes", "No");
			return n == 0;
		}

		static bool Quit()
		{
			var n = MessageBox.Query(50, 7, "Quit GitCli", "Are you sure you want to quit this GitCli?", "Yes", "No");
			return n == 0;
		}

		void OpenRepository()
		{
			using (var repo = new Repository("D:/VDisk/Github/Samples"))
			{
				//var master = repo.Branches["master"];
				var status = repo.RetrieveStatus();
				//status.IsDirty;

				//status.Modified
			}
		}
	}
}
