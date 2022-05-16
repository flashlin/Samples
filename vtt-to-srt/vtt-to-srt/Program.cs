using System.Xml.Linq;
using vtt_to_srt;


var vttFile = args[0];
var vttFileContent = File.ReadAllText(vttFile);
var xml = XDocument.Parse(vttFileContent);


var index = 1;
var currentTime = new TimeSpan(0, 0, 0);

//1
//00:00:00,900-- > 00:00:02,402
var path = Path.GetDirectoryName(vttFile);
var fileName = Path.GetFileNameWithoutExtension(vttFile);
var srtFile = Path.Combine(path, fileName + ".srt");

using var srtStream = new FileStream(srtFile, FileMode.Create);
using var sw = new StreamWriter(srtStream);

foreach (var xText in xml.Root.Elements("text"))
{
	var startSecs = xText.Attribute("start").Value;
	var dur = xText.Attribute("dur").Value;
	var text = xText.Value;

	var startSpan = startSecs.ToTimeSpan();
	var durSpan = dur.ToTimeSpan();
	var endSpan = startSpan + durSpan;

	var startSrt = startSpan.ToSrtFormat();
	var endSrt = endSpan.ToSrtFormat();

	sw.WriteLine(index);
	sw.WriteLine($"{startSrt} --> {endSrt}");
	sw.WriteLine(text);
	sw.WriteLine();
}

sw.Flush();