namespace ChargeLimitConfig_DesignPattern1
{
	public class ChargeLimit
	{
		public int PeriodDays { get; set; }
		public int Amount { get; set; }

		public bool IsUnlimit => Amount == UnlimitAmount;
		public static int UnlimitAmount = 0;
	}
}