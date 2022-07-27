// See https://aka.ms/new-console-template for more information

Console.WriteLine("Hello, Screen Recorder!");
var screenRecorder = new ScreenRecorder();
screenRecorder.Start();
Console.ReadLine();
screenRecorder.Stop();