using System;
using T1.Standard.DesignPatterns;

namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public class CannotIncreaseRule : ChainOfResponsibilityHandler<ValidateChargeLimitArgs>, IChargeLimitUpdateRule
	{
		public override void Handle(ValidateChargeLimitArgs args)
		{
			if (args.OldLimit.Amount < args.NewLimit.Amount)
			{
				throw new ValidationException();
			}
		}
	}
}