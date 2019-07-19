using System;
using T1.Standard.DesignPatterns;

namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public class CannotLimitToUnlimitRule : ChainOfResponsibilityHandler<ValidateChargeLimitArgs>, IChargeLimitUpdateRule
	{
		public override void Handle(ValidateChargeLimitArgs args)
		{
			if (!args.OldLimit.IsUnlimit && args.NewLimit.IsUnlimit )
			{
				throw new Exception();
			}
		}
	}
}