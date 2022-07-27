using ScreenRecorderLib;

public class ScreenRecorder
{
	Recorder _rec;

	public void Start()
	{
		//string videoPath = Path.Combine(Path.GetTempPath(), "test.mp4");
		var videoPath = "d:/demo/test.mp4";

		var options = new RecorderOptions
		{
			SourceOptions = new SourceOptions
			{
				RecordingSources = new List<RecordingSourceBase>()
				{
					//new DisplayRecordingSource(@"\\.\DISPLAY2")
					new DisplayRecordingSource(DisplayRecordingSource.MainMonitor)
				}
			},
			OutputOptions = new OutputOptions
			{
				RecorderMode = RecorderMode.Video,
				OutputFrameSize = new ScreenSize(1280, 720),
				//Stretch controls how the resizing is done, if the new aspect ratio differs.
				Stretch = StretchMode.Uniform,
				//SourceRect = new ScreenRect(100, 100, 500, 500)
			},
			AudioOptions = new AudioOptions
			{
				Bitrate = AudioBitrate.bitrate_128kbps,
				Channels = AudioChannels.Stereo,
				IsAudioEnabled = true,
			},
			VideoEncoderOptions = new VideoEncoderOptions
			{
				Bitrate = 8000 * 1000,
				Framerate = 15,
				IsFixedFramerate = true,
				Encoder = new H264VideoEncoder
				{
					BitrateMode = H264BitrateControlMode.CBR,
					EncoderProfile = H264Profile.Main,
				},
				IsFragmentedMp4Enabled = true,
				IsThrottlingDisabled = false,
				IsHardwareEncodingEnabled = true,
				IsLowLatencyEnabled = false,
				IsMp4FastStartEnabled = true
			},
		};

		_rec = Recorder.CreateRecorder(options);
		_rec.OnRecordingComplete += OnRecordingComplete;
		_rec.OnRecordingFailed += OnRecordingFailed;
		_rec.OnStatusChanged += OnStatusChanged;
		_rec.Record(videoPath);
	}

	public void Stop()
	{
		_rec.Stop();
	}

	private void OnRecordingComplete(object? sender, RecordingCompleteEventArgs e)
	{
		var path = e.FilePath;
	}

	private void OnRecordingFailed(object? sender, RecordingFailedEventArgs e)
	{
		var error = e.Error;
	}

	private void OnStatusChanged(object? sender, RecordingStatusEventArgs e)
	{
		var status = e.Status;
	}
}