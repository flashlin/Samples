using System;

namespace ChargeLimitConfig_DesignPattern1
{
	public class TimeHelper
	{
		public static bool IsIn24Hr(DateTime lastModifiedTime, DateTime nowTime)
		{
			return nowTime - lastModifiedTime < TimeSpan.FromHours(24);
		}
	}
}