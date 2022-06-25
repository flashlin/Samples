using GitMaui.Models;
using LibGit2Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitMaui.Services
{
	public class GitService
	{

		public IEnumerable<GitFileInfo> QueryStatus(string folderPath)
		{
			using (var repo = new Repository(folderPath))
			{
				//var master = repo.Branches["master"];
				var status = repo.RetrieveStatus();
				//status.IsDirty;
				//status.Modified
				foreach (var modified in status.Modified)
				{
					yield return new GitFileInfo
					{
						FilePath = modified.FilePath,
						Status = GitFileStatus.Modified
					};
				}
			}
		}
	}
}
