using System;
using System.Diagnostics;
using System.Text;

namespace T1.SqlLocalData
{
	public class ProcessHelper
	{
		public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);

		public string Execute(string processFilename, string arguments)
		{
			var startInfo = new ProcessStartInfo(processFilename)
			{
				WindowStyle = ProcessWindowStyle.Hidden,
				Arguments = arguments,
				RedirectStandardError = true,
				RedirectStandardOutput = true,
				CreateNoWindow = true,
				UseShellExecute = false,
			};

			var outputStringBuilder = new StringBuilder();

			var process = new Process();
			process.StartInfo = startInfo;
			process.EnableRaisingEvents = false;
			process.OutputDataReceived += (sender, eventArgs) => outputStringBuilder.AppendLine(eventArgs.Data);
			process.ErrorDataReceived += (sender, eventArgs) => outputStringBuilder.AppendLine(eventArgs.Data);

			try
			{
				process.Start();
				process.BeginOutputReadLine();
				process.BeginErrorReadLine();
				//process.StandardOutput.ReadToEnd();
				var processExited = process.WaitForExit((int)Timeout.TotalMilliseconds);
				if (processExited == false)
				{
					process.Kill();
					throw new Exception("ERROR: Process took too long to finish");
				}

				if (process.ExitCode != 0)
				{
					var commandLine = $"{processFilename} {arguments}";
					var output = outputStringBuilder.ToString();
					throw new Exception($"{commandLine}" + Environment.NewLine
						+ "Process exited code: " + process.ExitCode + Environment.NewLine
						+ "Output from process: " + outputStringBuilder.ToString());
				}
			}
			finally
			{
				process.Close();
			}

			return outputStringBuilder.ToString();
		}
	}
}