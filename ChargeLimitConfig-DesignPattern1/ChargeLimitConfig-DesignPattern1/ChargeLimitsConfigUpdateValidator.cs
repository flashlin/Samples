﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ChargeLimitConfig_DesignPattern1.Rules;
using T1.Standard.DesignPatterns;

namespace ChargeLimitConfig_DesignPattern1
{
	public class ChargeLimitsConfigUpdateValidator
	{
		public bool Validate(ChargeLimitsConfig oldConfig, ChargeLimitsConfig newConfig)
		{
			var allChargeLimits = GetAllChargeLimits(oldConfig, newConfig)
				.ToList();

			if (!IsNoAnyChanged(allChargeLimits))
			{
				return false;
			}

			var rule = ChainOfResponsibilityHandler.Chain(
				new In24HrRule(),
				new Out24HrRule());

			return HandleAllChargeLimitsByRule(allChargeLimits, rule);
		}


		private static bool IsNoAnyChanged(IEnumerable<ValidateChargeLimitArgs> allChargeLimits)
		{
			var validateChargeLimitArgses = allChargeLimits as ValidateChargeLimitArgs[] ?? allChargeLimits.ToArray();

			var equalsCount = validateChargeLimitArgses
				.Count(x => x.OldLimit.Amount == x.NewLimit.Amount);

			return validateChargeLimitArgses.Count() == equalsCount;
		}

		private static IEnumerable<ValidateChargeLimitArgs> GetAllChargeLimits(ChargeLimitsConfig oldConfig, ChargeLimitsConfig newConfig)
		{
			var q1 = from tb1 in oldConfig.PeriodDayLimits.Values
						join tb2 in newConfig.PeriodDayLimits.Values on tb1.PeriodDays equals tb2.PeriodDays
						select new ValidateChargeLimitArgs()
						{
							OldLimit = tb1,
							LastModifiedTime = oldConfig.LastModifiedTime,
							NewLimit = tb2,
							ModifyTime = newConfig.LastModifiedTime
						};
			return q1;
		}

		private static bool HandleAllChargeLimitsByRule(IEnumerable<ValidateChargeLimitArgs> chargeLimits, IChainOfResponsibilityHandler<ValidateChargeLimitArgs> rule)
		{
			try
			{
				foreach (var limit in chargeLimits)
				{
					rule.Handle(limit);
				}
				return true;
			}
			catch(ValidationException)
			{
				return false;
			}
		}
	}
}
