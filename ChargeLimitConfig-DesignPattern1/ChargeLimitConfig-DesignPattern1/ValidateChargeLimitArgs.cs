using System;

namespace ChargeLimitConfig_DesignPattern1
{
	public class ValidateChargeLimitArgs
	{
		public DateTime LastModifiedTime { get; set; }
		public DateTime ModifyTime { get; set; }
		public ChargeLimit NewLimit { get; set; }
		public ChargeLimit OldLimit { get; set; }
	}
}