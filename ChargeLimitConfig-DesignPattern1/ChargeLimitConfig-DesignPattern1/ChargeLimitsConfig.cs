using System;
using System.Collections.Generic;

namespace ChargeLimitConfig_DesignPattern1
{
	public class ChargeLimitsConfig
	{
		public Dictionary<int, ChargeLimit> PeriodDayLimits { get; set; } = new Dictionary<int, ChargeLimit>();
		public DateTime LastModifiedTime { get; set; }
	}
}