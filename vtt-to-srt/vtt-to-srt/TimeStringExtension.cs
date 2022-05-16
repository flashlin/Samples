using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vtt_to_srt
{
	public static class TimeStringExtension
	{
		public static TimeSpan ToTimeSpan(this string sec)
		{
			var milli = float.Parse(sec) * 1000;
			return TimeSpan.FromMilliseconds(milli);
		}

		public static string ToSrtFormat(this TimeSpan t)
		{
			var hour = t.Hours.ToString("00");
			var min = t.Minutes.ToString("00");
			var sec = t.Seconds.ToString("00");
			var milli = t.Milliseconds.ToString("000");
			return $"{hour}:{min}:{sec},{milli}";
		}
	}
}
